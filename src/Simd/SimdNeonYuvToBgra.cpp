/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> SIMD_INLINE void YuvToBgra(const uint8x16_t & y, const uint8x16_t & u, const uint8x16_t & v, const uint8x16_t & a, uint8_t * bgra)
        {
            uint8x16x4_t _bgra;
            YuvToBgr(y, u, v, *(uint8x16x3_t*)&_bgra);
            _bgra.val[3] = a;
            Store4<align>(bgra, _bgra);
        }

        template <bool align> SIMD_INLINE void Yuva422pToBgra(const uint8_t * y, const uint8x16x2_t & u, const uint8x16x2_t & v, const uint8_t * a, uint8_t * bgra)
        {
            YuvToBgra<align>(Load<align>(y + 0), u.val[0], v.val[0], Load<align>(a + 0), bgra + 0);
            YuvToBgra<align>(Load<align>(y + A), u.val[1], v.val[1], Load<align>(a + A), bgra + QA);
        }

        template <bool align> void Yuva420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));
                assert(Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuva422pToBgra<align>(y + colY, _u, _v, a + colY, bgra + colBgra);
                    Yuva422pToBgra<align>(y + colY + yStride, _u, _v, a + colY + aStride, bgra + colBgra + bgraStride);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + offset / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + offset / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuva422pToBgra<false>(y + offset, _u, _v, a + offset, bgra + 4 * offset);
                    Yuva422pToBgra<false>(y + offset + yStride, _u, _v, a + offset + aStride, bgra + 4 * offset + bgraStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuva420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride)
                && Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuva420pToBgra<true>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
            else
                Yuva420pToBgra<false>(y, yStride, u, uStride, v, vStride, a, aStride, width, height, bgra, bgraStride);
        }

        template <bool align> SIMD_INLINE void Yuv422pToBgra(const uint8_t * y, const uint8x16x2_t & u, const uint8x16x2_t & v, const uint8x16_t & alpha, uint8_t * bgra)
        {
            YuvToBgra<align>(Load<align>(y + 0), u.val[0], v.val[0], alpha, bgra + 0);
            YuvToBgra<align>(Load<align>(y + A), u.val[1], v.val[1], alpha, bgra + QA);
        }

        template <bool align> void Yuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            uint8x16_t _alpha = vdupq_n_u8(alpha);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToBgra<align>(y + colY, _u, _v, _alpha, bgra + colBgra);
                    Yuv422pToBgra<align>(y + colY + yStride, _u, _v, _alpha, bgra + colBgra + bgraStride);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + offset / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + offset / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToBgra<false>(y + offset, _u, _v, _alpha, bgra + 4 * offset);
                    Yuv422pToBgra<false>(y + offset + yStride, _u, _v, _alpha, bgra + 4 * offset + bgraStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void Yuv420pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv420pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv420pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }

        template <bool align> void Yuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            uint8x16_t _alpha = vdupq_n_u8(alpha);
            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToBgra<align>(y + colY, _u, _v, _alpha, bgra + colBgra);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + offset / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + offset / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToBgra<false>(y + offset, _u, _v, _alpha, bgra + 4 * offset);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv422pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv422pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv422pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }

        template <bool align> void Yuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            uint8x16_t _alpha = vdupq_n_u8(alpha);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < bodyWidth; col += A, colBgra += QA)
                {
                    uint8x16_t _y = Load<align>(y + col);
                    uint8x16_t _u = Load<align>(u + col);
                    uint8x16_t _v = Load<align>(v + col);
                    YuvToBgra<align>(_y, _u, _v, _alpha, bgra + colBgra);
                }
                if (tail)
                {
                    size_t col = width - A;
                    uint8x16_t _y = Load<false>(y + col);
                    uint8x16_t _u = Load<false>(u + col);
                    uint8x16_t _v = Load<false>(v + col);
                    YuvToBgra<false>(_y, _u, _v, _alpha, bgra + 4 * col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void Yuv444pToBgra(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv444pToBgra<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
            else
                Yuv444pToBgra<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha);
        }

        //-----------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void YuvToBgra(const uint8x16_t& y, const uint8x16_t& u, const uint8x16_t& v, const uint8x16_t& a, uint8_t* bgra)
        {
            uint8x16x4_t _bgra;
            YuvToBgr<T>(y, u, v, *(uint8x16x3_t*)&_bgra);
            _bgra.val[3] = a;
            Store4<align>(bgra, _bgra);
        }

        template <class T, bool align> SIMD_INLINE void Yuva422pToBgra(const uint8_t* y, const uint8x16x2_t& u, const uint8x16x2_t& v, const uint8x16_t& a, uint8_t* bgra)
        {
            YuvToBgra<T, align>(Load<align>(y + 0), u.val[0], v.val[0], a, bgra + 0);
            YuvToBgra<T, align>(Load<align>(y + A), u.val[1], v.val[1], a, bgra + QA);
        }

        template <bool align, class T> void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));
                assert(Aligned(bgra) && Aligned(bgraStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16_t _a = vdupq_n_u8(alpha);
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < bodyWidth; colY += DA, colUV += A, colBgra += OA)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuva422pToBgra<T, align>(y + colY, _u, _v, _a, bgra + colBgra);
                    Yuva422pToBgra<T, align>(y + colY + yStride, _u, _v, _a, bgra + colBgra + bgraStride);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + offset / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + offset / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuva422pToBgra<T, false>(y + offset, _u, _v, _a, bgra + 4 * offset);
                    Yuva422pToBgra<T, false>(y + offset + yStride, _u, _v, _a, bgra + 4 * offset + bgraStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        template <bool align> void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToBgraV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv420pToBgraV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv420pToBgraV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv420pToBgraV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        void Yuv420pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv420pToBgraV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
            else
                Yuv420pToBgraV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, class T> void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            uint8x16_t _alpha = vdupq_n_u8(alpha);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < bodyWidth; col += A, colBgra += QA)
                {
                    uint8x16_t _y = Load<align>(y + col);
                    uint8x16_t _u = Load<align>(u + col);
                    uint8x16_t _v = Load<align>(v + col);
                    YuvToBgra<T, align>(_y, _u, _v, _alpha, bgra + colBgra);
                }
                if (tail)
                {
                    size_t col = width - A;
                    uint8x16_t _y = Load<false>(y + col);
                    uint8x16_t _u = Load<false>(u + col);
                    uint8x16_t _v = Load<false>(v + col);
                    YuvToBgra<T, false>(_y, _u, _v, _alpha, bgra + 4 * col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        template <bool align> void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToBgraV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt709: Yuv444pToBgraV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvBt2020: Yuv444pToBgraV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            case SimdYuvTrect871: Yuv444pToBgraV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha); break;
            default:
                assert(0);
            }
        }

        void Yuv444pToBgraV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                Yuv444pToBgraV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
            else
                Yuv444pToBgraV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgra, bgraStride, alpha, yuvType);
        }
    }
#endif// SIMD_NEON_ENABLE
}

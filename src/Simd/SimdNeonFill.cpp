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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void FillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            size_t size = width * 3;
            size_t step = A * 3;
            size_t alignedSize = AlignLo(width, A) * 3;

            uint8x16x3_t bgr;
            bgr.val[0] = vdupq_n_u8(blue);
            bgr.val[1] = vdupq_n_u8(green);
            bgr.val[2] = vdupq_n_u8(red);

            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < alignedSize; offset += step)
                    Store3<align>(dst + offset, bgr);
                if (offset < size)
                    Store3<false>(dst + size - step, bgr);
                dst += stride;
            }
        }

        void FillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            if (Aligned(dst) && Aligned(stride))
                FillBgr<true>(dst, stride, width, height, blue, green, red);
            else
                FillBgr<false>(dst, stride, width, height, blue, green, red);
        }

        template <bool align> void FillBgra(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
        {
            size_t size = width * 4;
            size_t alignedSize = AlignLo(width, A) * 4;

            uint8x16x4_t bgra;
            bgra.val[0] = vdupq_n_u8(blue);
            bgra.val[1] = vdupq_n_u8(green);
            bgra.val[2] = vdupq_n_u8(red);
            bgra.val[3] = vdupq_n_u8(alpha);

            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < alignedSize; offset += QA)
                    Store4<align>(dst + offset, bgra);
                if (offset < size)
                    Store4<false>(dst + size - QA, bgra);
                dst += stride;
            }
        }

        void FillBgra(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
        {
            if (Aligned(dst) && Aligned(stride))
                FillBgra<true>(dst, stride, width, height, blue, green, red, alpha);
            else
                FillBgra<false>(dst, stride, width, height, blue, green, red, alpha);
        }

        template <bool align> void Fill32f(float * dst, size_t size, const float * value)
        {
            if (value == 0 || value[0] == 0)
                memset(dst, 0, size * sizeof(float));
            else
            {
                float v = value[0];
                const float * nose = (float*)AlignHi(dst, F * sizeof(float));
                for (; dst < nose && size; --size)
                    *dst++ = v;
                const float * end = dst + size;
                const float * endF = dst + AlignLo(size, F);
                const float * endQF = dst + AlignLo(size, QF);
                float32x4_t _v = vdupq_n_f32(v);
                for (; dst < endQF; dst += QF)
                {
                    Store<align>(dst + 0 * F, _v);
                    Store<align>(dst + 1 * F, _v);
                    Store<align>(dst + 2 * F, _v);
                    Store<align>(dst + 3 * F, _v);
                }
                for (; dst < endF; dst += F)
                    Store<align>(dst, _v);
                for (; dst < end;)
                    *dst++ = v;
            }
        }

        void Fill32f(float * dst, size_t size, const float * value)
        {
            if (Aligned(dst))
                Fill32f<true>(dst, size, value);
            else
                Fill32f<false>(dst, size, value);
        }

        template <bool align> void FillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8x16_t & pixel)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(dst) && Aligned(stride));

            size_t fullAlignedWidth = AlignLo(width, QA);
            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                {
                    Store<align>((dst + col) + 0 * A, pixel);
                    Store<align>((dst + col) + 1 * A, pixel);
                    Store<align>((dst + col) + 2 * A, pixel);
                    Store<align>((dst + col) + 3 * A, pixel);
                }
                for (; col < alignedWidth; col += A)
                    Store<align>((dst + col), pixel);
                if (col < width)
                    Store<false>((dst + width - A), pixel);
                dst += stride;
            }
        }

        template <bool align> void FillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize)
        {
            if (pixelSize == 3)
                FillBgr<align>(dst, stride, width, height, pixel[0], pixel[1], pixel[2]);
            else if (pixelSize == 1)
                Base::Fill(dst, stride, width, height, 1, pixel[0]);
            else
            {
                uint8x16_t _pixel;
                switch (pixelSize)
                {
                case 2:
                    _pixel = (uint8x16_t)vdupq_n_u16(*(uint16_t*)pixel);
                    break;
                case 4:
                    _pixel = (uint8x16_t)vdupq_n_u32(*(uint32_t*)pixel);
                    break;
                default:
                    assert(0);
                }
                FillPixel<align>(dst, stride, width*pixelSize, height, _pixel);
            }
        }

        void FillPixel(uint8_t * dst, size_t stride, size_t width, size_t height, const uint8_t * pixel, size_t pixelSize)
        {
            if (Aligned(dst) && Aligned(stride))
                FillPixel<true>(dst, stride, width, height, pixel, pixelSize);
            else
                FillPixel<false>(dst, stride, width, height, pixel, pixelSize);
        }
    }
#endif// SIMD_SSE2_ENABLE
}

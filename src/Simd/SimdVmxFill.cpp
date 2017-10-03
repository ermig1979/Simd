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
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template <bool align> void FillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            if (align)
                assert(Aligned(dst) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);

            v128_u8 bgr0 = SIMD_VEC_SETR_EPI8(blue, green, red, blue, green, red, blue, green, red, blue, green, red, blue, green, red, blue);
            v128_u8 bgr1 = SIMD_VEC_SETR_EPI8(green, red, blue, green, red, blue, green, red, blue, green, red, blue, green, red, blue, green);
            v128_u8 bgr2 = SIMD_VEC_SETR_EPI8(red, blue, green, red, blue, green, red, blue, green, red, blue, green, red, blue, green, red);

            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _dst(dst);
                Store<align, true>(_dst, bgr0);
                Store<align, false>(_dst, bgr1);
                Store<align, false>(_dst, bgr2);
                for (size_t col = A; col < alignedWidth; col += A)
                {
                    Store<align, false>(_dst, bgr0);
                    Store<align, false>(_dst, bgr1);
                    Store<align, false>(_dst, bgr2);
                }
                Flush(_dst);

                if (alignedWidth != width)
                {
                    Storer<false> _dst(dst + (width - A) * 3);
                    Store<false, true>(_dst, bgr0);
                    Store<false, false>(_dst, bgr1);
                    Store<false, false>(_dst, bgr2);
                    Flush(_dst);
                }

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
            if (align)
                assert(Aligned(dst) && Aligned(stride));

            uint32_t bgra32 = uint32_t(alpha) | (uint32_t(red) << 8) | (uint32_t(green) << 16) | (uint32_t(blue) << 24);

            size_t alignedWidth = AlignLo(width, 4);
            v128_u8 bgra128 = (v128_u8)SetU32(bgra32);
            for (size_t row = 0; row < height; ++row)
            {
                Storer<align> _dst(dst);
                Store<align, true>(_dst, bgra128);
                for (size_t col = 4; col < alignedWidth; col += 4)
                    Store<align, false>(_dst, bgra128);
                Flush(_dst);
                if (width != alignedWidth)
                    Store<false>(dst + 4 * (width - 4), bgra128);
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
    }
#endif// SIMD_VMX_ENABLE
}

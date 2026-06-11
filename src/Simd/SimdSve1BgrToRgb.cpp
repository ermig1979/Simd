/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2022-2022 Fabien Spindler,
*               2022-2022 Souriya Trinh.
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
#include "Simd/SimdSve1.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#ifdef SIMD_SVE_ENABLE
    namespace Sve
    {
        SIMD_INLINE void BgrToRgb(const uint8_t* bgr, uint8_t* rgb, const svbool_t & mask)
        {
            svuint8x3_t _bgr = svld3_u8(mask, bgr);
            svuint8x3_t _rgb = svcreate3_u8(svget3(_bgr, 2), svget3(_bgr, 1), svget3(_bgr, 0));
            svst3_u8(mask, rgb, _rgb);
        }

        void BgrToRgb(const uint8_t* bgr, size_t width, size_t height, size_t bgrStride, uint8_t* rgb, size_t rgbStride)
        {
            size_t A = svlen(svuint8_t()), A3 = A * 3;
            size_t widthA = AlignLo(width, A), size = width * 3, sizeA = widthA * 3;
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < sizeA; offset += A3)
                    BgrToRgb(bgr + offset, rgb + offset, body);
                if (widthA < width)
                    BgrToRgb(bgr + offset, rgb + offset, tail);
                bgr += bgrStride;
                rgb += rgbStride;
            }
        }
    }
#endif
}

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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint8_t TextureBoostedSaturatedGradient(const svuint8_t& a, const svuint8_t& b,
            const svuint8_t& saturation, const svuint8_t& boost, const svbool_t& mask)
        {
            svuint8_t positive = svmin_u8_x(mask, svqsub_u8_x(mask, b, a), saturation);
            svuint8_t negative = svmin_u8_x(mask, svqsub_u8_x(mask, a, b), saturation);
            return svmul_u8_x(mask, svsub_u8_x(mask, svadd_u8_x(mask, saturation, positive), negative), boost);
        }

        SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t* src, size_t stride, uint8_t* dx, uint8_t* dy,
            const svuint8_t& saturation, const svuint8_t& boost, const svbool_t& mask)
        {
            svuint8_t s10 = svld1_u8(mask, src - 1);
            svuint8_t s12 = svld1_u8(mask, src + 1);
            svuint8_t s01 = svld1_u8(mask, src - stride);
            svuint8_t s21 = svld1_u8(mask, src + stride);
            svst1_u8(mask, dx, TextureBoostedSaturatedGradient(s10, s12, saturation, boost, mask));
            svst1_u8(mask, dy, TextureBoostedSaturatedGradient(s01, s21, saturation, boost, mask));
        }

        void TextureBoostedSaturatedGradient(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t* dx, size_t dxStride, uint8_t* dy, size_t dyStride)
        {
            assert(int(2) * saturation * boost <= 0xFF);

            size_t A = svcntb();
            svuint8_t _saturation = svdup_n_u8(saturation);
            svuint8_t _boost = svdup_n_u8(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                dx[0] = 0;
                dy[0] = 0;
                for (size_t col = 1; col < width - 1; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, width - 1);
                    TextureBoostedSaturatedGradient(src + col, srcStride, dx + col, dy + col, _saturation, _boost, mask);
                }
                dx[width - 1] = 0;
                dy[width - 1] = 0;

                src += srcStride;
                dx += dxStride;
                dy += dyStride;
            }
            memset(dx, 0, width);
            memset(dy, 0, width);
        }
    }
#endif
}

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
        SIMD_INLINE svint16_t Half(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_s16_u16(svlsr_n_u16_x(mask, svld1_u16(mask, src), 1));
        }

        SIMD_INLINE void Anchor(const uint16_t* src, size_t stride, const svint16_t& threshold, uint8_t* dst, const svbool_t& mask)
        {
            svuint16_t _src = svld1_u16(mask, src);
            svuint16_t direction = svand_n_u16_x(mask, _src, 1);
            svuint16_t magnitude = svlsr_n_u16_x(mask, _src, 1);
            svint16_t value = svsub_s16_x(mask, svreinterpret_s16_u16(magnitude), threshold);

            svbool_t vertical = svand_b_z(mask, svcmpeq_n_u16(mask, direction, 1),
                svand_b_z(mask, svcmpge_s16(mask, value, Half(src - 1, mask)), svcmpge_s16(mask, value, Half(src + 1, mask))));
            svbool_t horizontal = svand_b_z(mask, svcmpeq_n_u16(mask, direction, 0),
                svand_b_z(mask, svcmpge_s16(mask, value, Half(src - stride, mask)), svcmpge_s16(mask, value, Half(src + stride, mask))));
            svbool_t anchor = svand_b_z(mask, svcmpgt_n_u16(mask, magnitude, 0), svorr_b_z(mask, vertical, horizontal));

            svst1b_u16(mask, dst, svsel_u16(anchor, svdup_n_u16(0xFF), svdup_n_u16(0)));
        }

        void ContourAnchors(const uint16_t* src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcnth();
            const svint16_t _threshold = svdup_n_s16(threshold);

            memset(dst, 0, width);
            memset(dst + dstStride * (height - 1), 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 1; row < height - 1; row += step)
            {
                dst[0] = 0;
                for (size_t col = 1; col < width - 1; col += A)
                    Anchor(src + col, srcStride, _threshold, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = 0;
                src += step * srcStride;
                dst += step * dstStride;
            }
        }

        void ContourAnchors(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t* dst, size_t dstStride)
        {
            assert(srcStride % sizeof(int16_t) == 0);

            ContourAnchors((const uint16_t*)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
        }
    }
#endif
}

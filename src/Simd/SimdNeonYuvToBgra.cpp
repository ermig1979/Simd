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
        template <class T, bool align> SIMD_INLINE void YuvToRgba(const uint8x16_t& y, const uint8x16_t& u, const uint8x16_t& v, const uint8x16_t& a, uint8_t* rgba)
        {
            uint8x16x4_t _rgba;
            YuvToRgba<T>(y, u, v, a, _rgba);
            Store4<align>(rgba, _rgba);
        }

        template <bool align, class T> void Yuv444pToRgbaV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(rgba) && Aligned(rgbaStride));
            }

            uint8x16_t _alpha = vdupq_n_u8(alpha);
            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colRgba = 0; col < bodyWidth; col += A, colRgba += QA)
                {
                    uint8x16_t _y = Load<align>(y + col);
                    uint8x16_t _u = Load<align>(u + col);
                    uint8x16_t _v = Load<align>(v + col);
                    YuvToRgba<T, align>(_y, _u, _v, _alpha, rgba + colRgba);
                }
                if (tail)
                {
                    size_t col = width - A;
                    uint8x16_t _y = Load<false>(y + col);
                    uint8x16_t _u = Load<false>(u + col);
                    uint8x16_t _v = Load<false>(v + col);
                    YuvToRgba<T, false>(_y, _u, _v, _alpha, rgba + 4 * col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                rgba += rgbaStride;
            }
        }

        template <bool align> void Yuv444pToRgbaV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToRgbaV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvBt709: Yuv444pToRgbaV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvBt2020: Yuv444pToRgbaV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            case SimdYuvTrect871: Yuv444pToRgbaV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha); break;
            default:
                assert(0);
            }
        }

        void Yuv444pToRgbaV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(rgba) && Aligned(rgbaStride))
                Yuv444pToRgbaV2<true>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha, yuvType);
            else
                Yuv444pToRgbaV2<false>(y, yStride, u, uStride, v, vStride, width, height, rgba, rgbaStride, alpha, yuvType);
        }
    }
#endif// SIMD_NEON_ENABLE
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
    namespace Base
    {
        template <class YuvType> SIMD_INLINE void Uyvy422ToBgr(const uint8_t* uyvy, uint8_t* bgr)
        {
            uint8_t u = uyvy[0], v = uyvy[2];
            YuvToBgr<YuvType>(uyvy[1], u, v, bgr + 0);
            YuvToBgr<YuvType>(uyvy[3], u, v, bgr + 3);
        }

        template <class YuvType> void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= 2));

            size_t sizeUyvy = width * 2;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUyvy = 0, colBgr = 0; colUyvy < sizeUyvy; colUyvy += 4, colBgr += 6)
                    Uyvy422ToBgr<YuvType>(uyvy + colUyvy, bgr + colBgr);
                uyvy += uyvyStride;
                bgr += bgrStride;
            }
        }

        void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Uyvy422ToBgr<Base::Bt601>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Uyvy422ToBgr<Base::Bt709>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Uyvy422ToBgr<Base::Bt2020>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Uyvy422ToBgr<Base::Trect871>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }
    }
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
        template <bool align, class T> SIMD_INLINE void Uyvy422ToBgr(const uint8_t* uyvy, uint8_t* bgr)
        {
            uint8x8x4_t uyvy0 = LoadHalf4<align>(uyvy);
            int16x8_t u = AdjustUV<T>(uyvy0.val[0]);
            int16x8_t y0 = AdjustY<T>(uyvy0.val[1]);
            int16x8_t v = AdjustUV<T>(uyvy0.val[2]);
            int16x8_t y1 = AdjustY<T>(uyvy0.val[3]);
            uint8x16x3_t _bgr;
            _bgr.val[0] = Combine(vzip_u8(vqmovun_s16(YuvToBlue<T>(y0, u)), vqmovun_s16(YuvToBlue<T>(y1, u))));
            _bgr.val[1] = Combine(vzip_u8(vqmovun_s16(YuvToGreen<T>(y0, u, v)), vqmovun_s16(YuvToGreen<T>(y1, u, v))));
            _bgr.val[2] = Combine(vzip_u8(vqmovun_s16(YuvToRed<T>(y0, v)), vqmovun_s16(YuvToRed<T>(y1, v))));
            Store3<align>(bgr, _bgr);
        }

        template <bool align, class T> void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= A));
            if (align)
                assert(Aligned(uyvy) && Aligned(uyvyStride) && Aligned(bgr) && Aligned(bgrStride));

            size_t sizeS = width * 2, sizeD = width * 3;
            size_t sizeS2A = AlignLo(sizeS, 2 * A);
            size_t tailS = sizeS - 2 * A;
            size_t tailD = sizeD - 3 * A;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colS = 0, colD = 0; colS < sizeS2A; colS += 2 * A, colD += 3 * A)
                    Uyvy422ToBgr<align, T>(uyvy + colS, bgr + colD);
                if (sizeS2A != sizeS)
                    Uyvy422ToBgr<false, T>(uyvy + tailS, bgr + tailD);
                uyvy += uyvyStride;
                bgr += bgrStride;
            }
}

        template<bool align> void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Uyvy422ToBgr<align, Base::Bt601>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Uyvy422ToBgr<align, Base::Bt709>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Uyvy422ToBgr<align, Base::Bt2020>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Uyvy422ToBgr<align, Base::Trect871>(uyvy, uyvyStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        void Uyvy422ToBgr(const uint8_t* uyvy, size_t uyvyStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            if (Aligned(uyvy) && Aligned(uyvyStride) && Aligned(bgr) && Aligned(bgrStride))
                Uyvy422ToBgr<true>(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
            else
                Uyvy422ToBgr<false>(uyvy, uyvyStride, width, height, bgr, bgrStride, yuvType);
        }
    }
#endif
}

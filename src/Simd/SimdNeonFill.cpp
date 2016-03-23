/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> void FillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            size_t size = width*3;
            size_t step = A*3;
            size_t alignedSize = AlignLo(width, A)*3;

            uint8x16x3_t bgr;
            bgr.val[0] = vdupq_n_u8(blue);
            bgr.val[1] = vdupq_n_u8(green);
            bgr.val[2] = vdupq_n_u8(red);

            for(size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for(; offset < alignedSize; offset += step)
                    Store3<align>(dst + offset, bgr);
                if(offset < size)
                    Store3<false>(dst + size - step, bgr);
                dst += stride;
            }
        }

        void FillBgr(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red)
        {
            if(Aligned(dst) && Aligned(stride))
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
            if(Aligned(dst) && Aligned(stride))
                FillBgra<true>(dst, stride, width, height, blue, green, red, alpha);
            else
                FillBgra<false>(dst, stride, width, height, blue, green, red, alpha);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
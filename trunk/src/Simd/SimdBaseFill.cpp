/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        void Fill(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uint8_t value)
        {
            size_t rowSize = width*pixelSize;
            for(size_t row = 0; row < height; ++row)
            {
                memset(dst, value, rowSize);
                dst += stride;
            }
        }

        void FillFrame(uint8_t * dst, size_t stride, size_t width, size_t height, size_t pixelSize, 
            size_t frameLeft, size_t frameTop, size_t frameRight, size_t frameBottom, uint8_t value)
        {
            if(frameTop)
            {
                size_t offset = 0;
                size_t size = width*pixelSize;
                for(size_t row = 0; row < frameTop; ++row)
                {
                    memset(dst + offset, value, size);
                    offset += stride;
                }
            }
            if(height - frameBottom)
            {
                size_t offset = frameBottom*stride;
                size_t size = width*pixelSize;
                for(size_t row = frameBottom; row < height; ++row)
                {
                    memset(dst + offset, value, size);
                    offset += stride;
                }
            }
            if(frameLeft)
            {
                size_t offset = frameTop*stride;
                size_t size = frameLeft*pixelSize;
                for(size_t row = frameTop; row < frameBottom; ++row)
                {
                    memset(dst + offset, value, size);
                    offset += stride;
                }
            }
            if(width - frameRight)
            {
                size_t offset = frameTop*stride + frameRight*pixelSize;
                size_t size = (width - frameRight)*pixelSize;
                for(size_t row = frameTop; row < frameBottom; ++row)
                {
                    memset(dst + offset, value, size);
                    offset += stride;
                }
            }
        }

        void FillBgra(uint8_t * dst, size_t stride, size_t width, size_t height, uint8_t blue, uint8_t green, uint8_t red, uint8_t alpha)
        {
            uint32_t bgra32 = uint32_t(blue) | (uint32_t(green) << 8) | (uint32_t(red) << 16) | (uint32_t(alpha) << 24);

#if (defined _MSC_VER && defined _M_X64) || (defined __GNUC__ && defined __x86_64__)
            uint64_t bgra64 = uint64_t(bgra32) | (uint64_t(bgra32) << 32);
            size_t alignedWidth = AlignLo(width, 2);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += 2)
                    *((uint64_t*)((uint32_t*)dst + col)) = bgra64;
                if(width != alignedWidth)
                    ((uint32_t*)dst)[width - 1] = bgra32;
                dst += stride;
            }
#else
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < width; ++col)
                    ((uint32_t*)dst)[col] = bgra32;
                dst += stride;
            }
#endif        
        }
    }
}
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
#include "Simd/SimdEnable.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdFill.h"

namespace Simd
{
    namespace Base
    {
        void FillBgra(uchar * dst, size_t stride, size_t width, size_t height, uchar blue, uchar green, uchar red, uchar alpha)
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

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool align> void FillBgra(uchar * dst, size_t stride, size_t width, size_t height, uchar blue, uchar green, uchar red, uchar alpha)
        {
            uint32_t bgra32 = uint32_t(blue) | (uint32_t(green) << 8) | (uint32_t(red) << 16) | (uint32_t(alpha) << 24);
            size_t alignedWidth = AlignLo(width, 4);
            __m128i bgra128 = _mm_set1_epi32(bgra32);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += 4)
                    Store<align>((__m128i*)((uint32_t*)dst + col), bgra128);
                if(width != alignedWidth)
                    Store<false>((__m128i*)((uint32_t*)dst + width - 4), bgra128);
                dst += stride;
            }
        }

        void FillBgra(uchar * dst, size_t stride, size_t width, size_t height, uchar blue, uchar green, uchar red, uchar alpha)
        {
            if(Aligned(dst) && Aligned(stride))
                FillBgra<true>(dst, stride, width, height, blue, green, red, alpha);
            else
                FillBgra<false>(dst, stride, width, height, blue, green, red, alpha);
        }
    }
#endif// SIMD_SSE2_ENABLE

    void Fill(uchar * dst, size_t stride, size_t width, size_t height, size_t pixelSize, uchar value)
    {
        size_t rowSize = width*pixelSize;
        for(size_t row = 0; row < height; ++row)
        {
            memset(dst, value, rowSize);
            dst += stride;
        }
    }

    void FillBgra(uchar * dst, size_t stride, size_t width, size_t height, uchar blue, uchar green, uchar red, uchar alpha)
    {
#ifdef SIMD_AVX2_ENABLE
        if(Avx2::Enable && width >= Avx2::A)
            Avx2::FillBgra(dst, stride, width, height, blue, green, red, alpha);
        else
#endif// SIMD_AVX2_ENABLE
#ifdef SIMD_SSE2_ENABLE
        if(Sse2::Enable && width >= Sse2::A)
            Sse2::FillBgra(dst, stride, width, height, blue, green, red, alpha);
        else
#endif// SIMD_SSE2_ENABLE
            Base::FillBgra(dst, stride, width, height, blue, green, red, alpha);
    }
}
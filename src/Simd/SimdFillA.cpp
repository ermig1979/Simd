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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> void FillBgra(uchar * dst, size_t stride, size_t width, size_t height, uchar blue, uchar green, uchar red, uchar alpha)
        {
            uint32_t bgra32 = uint32_t(blue) | (uint32_t(green) << 8) | (uint32_t(red) << 16) | (uint32_t(alpha) << 24);
            size_t alignedWidth = AlignLo(width, 8);
            __m256i bgra256 = _mm256_set1_epi32(bgra32);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += 8)
                    Store<align>((__m256i*)((uint32_t*)dst + col), bgra256);
                if(width != alignedWidth)
                    Store<false>((__m256i*)((uint32_t*)dst + width - 8), bgra256);
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
#endif// SIMD_AVX2_ENABLE
}
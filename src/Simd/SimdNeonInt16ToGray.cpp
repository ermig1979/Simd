/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> SIMD_INLINE void Int16ToGray(const int16_t * src, uint8_t * dst)
        {
            int16x8_t _src = Load<align>(src);
            Store<align>(dst, vqmovun_s16(_src));
        }

        template <bool align> void Int16ToGray(const int16_t * src, size_t width, size_t height, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(src) && Aligned(srcStride, HA) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, HA);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += HA)
                    Int16ToGray<align>(src + col, dst + col);
                if (alignedWidth != width)
                    Int16ToGray<false>(src + width - HA, dst + width - HA);
                src += srcStride;
                dst += dstStride;
            }
        }

        void Int16ToGray(const uint8_t * src, size_t width, size_t height, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                Int16ToGray<true>((const int16_t *)src, width, height, srcStride / sizeof(int16_t), dst, dstStride);
            else
                Int16ToGray<false>((const int16_t *)src, width, height, srcStride / sizeof(int16_t), dst, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}

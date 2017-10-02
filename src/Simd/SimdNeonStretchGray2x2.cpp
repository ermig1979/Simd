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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<bool align> SIMD_INLINE void StretchGray2x2(const uint8_t * src, uint8_t * dst, size_t stride)
        {
            uint8x16x2_t _src;
            _src.val[0] = Load<align>(src);
            _src.val[1] = _src.val[0];
            Store2<align>(dst, _src);
            Store2<align>(dst + stride, _src);
        }

        template <bool align> void StretchGray2x2(
            const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth * 2 == dstWidth && srcHeight * 2 == dstHeight && srcWidth >= A);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(srcWidth, A);
            for (size_t row = 0; row < srcHeight; ++row)
            {
                for (size_t srcCol = 0, dstCol = 0; srcCol < alignedWidth; srcCol += A, dstCol += DA)
                    StretchGray2x2<align>(src + srcCol, dst + dstCol, dstStride);
                if (alignedWidth != srcWidth)
                    StretchGray2x2<false>(src + srcWidth - A, dst + dstWidth - DA, dstStride);
                src += srcStride;
                dst += 2 * dstStride;
            }
        }

        void StretchGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                StretchGray2x2<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                StretchGray2x2<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}

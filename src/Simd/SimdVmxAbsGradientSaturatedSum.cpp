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
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template<bool align> SIMD_INLINE v128_u8 AbsGradientSaturatedSum(const uint8_t * src, size_t stride)
        {
            const v128_u8 dx = AbsDifferenceU8(Load<false>(src - 1), Load<false>(src + 1));
            const v128_u8 dy = AbsDifferenceU8(Load<align>(src - stride), Load<align>(src + stride));
            return vec_adds(dx, dy);
        }

        template<bool align> SIMD_INLINE void AbsGradientSaturatedSum(const uint8_t * src, size_t stride, uint8_t * dst, size_t offset)
        {
            Store<align>(dst + offset, AbsGradientSaturatedSum<align>(src + offset, stride));
        }

        template<bool align> void AbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            size_t fullAlignedWidth = AlignLo(width, QA);
            memset(dst, 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 2; row < height; ++row)
            {
                if (align)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QA)
                    {
                        AbsGradientSaturatedSum<align>(src, srcStride, dst, col);
                        AbsGradientSaturatedSum<align>(src, srcStride, dst, col + A);
                        AbsGradientSaturatedSum<align>(src, srcStride, dst, col + 2 * A);
                        AbsGradientSaturatedSum<align>(src, srcStride, dst, col + 3 * A);
                    }
                    for (; col < alignedWidth; col += A)
                        AbsGradientSaturatedSum<align>(src, srcStride, dst, col);
                }
                else
                {
                    Storer<align> _dst(dst);
                    _dst.First(AbsGradientSaturatedSum<align>(src, srcStride));
                    for (size_t col = A; col < alignedWidth; col += A)
                        _dst.Next(AbsGradientSaturatedSum<align>(src + col, srcStride));
                    Flush(_dst);
                }

                if (width != alignedWidth)
                    AbsGradientSaturatedSum<false>(src, srcStride, dst, width - A);

                dst[0] = 0;
                dst[width - 1] = 0;

                src += srcStride;
                dst += dstStride;
            }
            memset(dst, 0, width);
        }

        void AbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AbsGradientSaturatedSum<true>(src, srcStride, width, height, dst, dstStride);
            else
                AbsGradientSaturatedSum<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}

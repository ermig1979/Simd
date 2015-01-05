/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        template<bool align, bool first> 
        SIMD_INLINE void AbsGradientSaturatedSum(const uint8_t * src, size_t stride, Storer<align> & dst)
        {
            const v128_u8 s10 = Load<false>(src - 1);
            const v128_u8 s12 = Load<false>(src + 1);
            const v128_u8 s01 = Load<align>(src - stride);
            const v128_u8 s21 = Load<align>(src + stride);
            const v128_u8 dx = AbsDifferenceU8(s10, s12);
            const v128_u8 dy = AbsDifferenceU8(s01, s21);
            Store<align, first>(dst, vec_adds(dx, dy));
        }

        template<bool align> void AbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            memset(dst, 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 2; row < height; ++row)
            {
                Storer<align> _dst(dst);
                AbsGradientSaturatedSum<align, true>(src, srcStride, _dst);
                for (size_t col = A; col < alignedWidth; col += A)
                    AbsGradientSaturatedSum<align, false>(src + col, srcStride, _dst);
                Flush(_dst);

                if(width != alignedWidth)
                {
                    Storer<false> _dst(dst + width - A);
                    AbsGradientSaturatedSum<false, true>(src + width - A, srcStride, _dst);
                    Flush(_dst);
                }

                dst[0] = 0;
                dst[width - 1] = 0;

                src += srcStride;
                dst += dstStride;
            }
            memset(dst, 0, width);
        }

        void AbsGradientSaturatedSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AbsGradientSaturatedSum<true>(src, srcStride, width, height, dst, dstStride);
            else
                AbsGradientSaturatedSum<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif// SIMD_VSX_ENABLE
}
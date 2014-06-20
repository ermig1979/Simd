/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
        template <bool align> void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);

            memset(sums, 0, sizeof(uint32_t)*height);
            const uint8_t * src0 = src;
            const uint8_t * src1 = src + stride;
            height--;
            for(size_t row = 0; row < height; ++row)
            {
                v128_u32 sum = K32_00000000;
                for(size_t col = 0; col < alignedWidth; col += A)
                {
                    v128_u8 _src0 = Load<align>(src0 + col);
                    v128_u8 _src1 = Load<align>(src1 + col);
                    sum = vec_msum(AbsDifferenceU8(_src0, _src1), K8_01, sum);
                }
                if(alignedWidth != width)
                {
                    v128_u8 _src0 = Load<false>(src0 + width - A);
                    v128_u8 _src1 = Load<false>(src1 + width - A);
                    sum = vec_msum(AbsDifferenceU8(_src0, _src1), tailMask, sum);
                }
                sums[row] = ExtractSum(sum);
                src0 += stride;
                src1 += stride;
            }
        }

        void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if(Aligned(src) && Aligned(stride))
                GetAbsDyRowSums<true>(src, stride, width, height, sums);
            else
                GetAbsDyRowSums<false>(src, stride, width, height, sums);
        }
    }
#endif// SIMD_VSX_ENABLE
}
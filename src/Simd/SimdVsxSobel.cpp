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
        SIMD_INLINE void SobelDxAbs(v128_u8 a[3][3], v128_u16 & lo, v128_u16 & hi)
        {
            lo = (v128_u16)vec_abs((v128_s16)BinomialSum(
                vec_sub(UnpackLoU8(a[0][2]), UnpackLoU8(a[0][0])),
                vec_sub(UnpackLoU8(a[1][2]), UnpackLoU8(a[1][0])),
                vec_sub(UnpackLoU8(a[2][2]), UnpackLoU8(a[2][0])))); 
            hi = (v128_u16)vec_abs((v128_s16)BinomialSum(
                vec_sub(UnpackHiU8(a[0][2]), UnpackHiU8(a[0][0])),
                vec_sub(UnpackHiU8(a[1][2]), UnpackHiU8(a[1][0])),
                vec_sub(UnpackHiU8(a[2][2]), UnpackHiU8(a[2][0])))); 
        }

        SIMD_INLINE void SobelDyAbs(v128_u8 a[3][3], v128_u16 & lo, v128_u16 & hi)
        {
            lo = (v128_u16)vec_abs((v128_s16)BinomialSum(
                vec_sub(UnpackLoU8(a[2][0]), UnpackLoU8(a[0][0])),
                vec_sub(UnpackLoU8(a[2][1]), UnpackLoU8(a[0][1])),
                vec_sub(UnpackLoU8(a[2][2]), UnpackLoU8(a[0][2])))); 
            hi = (v128_u16)vec_abs((v128_s16)BinomialSum(
                vec_sub(UnpackHiU8(a[2][0]), UnpackHiU8(a[0][0])),
                vec_sub(UnpackHiU8(a[2][1]), UnpackHiU8(a[0][1])),
                vec_sub(UnpackHiU8(a[2][2]), UnpackHiU8(a[0][2])))); 
        }

        SIMD_INLINE v128_u16 ContourMetrics(v128_u16 dx, v128_u16 dy)
        {
            return vec_or(vec_sl(vec_add(dx, dy), K16_0001), vec_and(vec_cmplt(dx, dy), K16_0001)); 
        }

        SIMD_INLINE void ContourMetrics(v128_u8 a[3][3], v128_u16 & lo, v128_u16 & hi)
        {
            v128_u16 dxLo, dxHi, dyLo, dyHi;
            SobelDxAbs(a, dxLo, dxHi);
            SobelDyAbs(a, dyLo, dyHi);
            lo = ContourMetrics(dxLo, dyLo);
            hi = ContourMetrics(dxHi, dyHi);
        }

        template<bool align, bool first> SIMD_INLINE void ContourMetrics(v128_u8 a[3][3], Storer<align> & dst)
        {
            v128_u16 lo, hi;
            ContourMetrics(a, lo, hi);
            Store<align, first>(dst, lo); 
            Store<align, false>(dst, hi); 
        }

        template <bool align> void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Sse2::Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            v128_u8 a[3][3];

            for(size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if(row == 0)
                    src0 = src1;
                if(row == height - 1)
                    src2 = src1;

                Storer<align> _dst(dst);
                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                ContourMetrics<align, true>(a, _dst);
                for(size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    ContourMetrics<align, false>(a, _dst);
                }
                _dst.Flush();

                {
                    Storer<false> _dst(dst + width - A);
                    LoadTail3<false, 1>(src0 + width - A, a[0]);
                    LoadTail3<false, 1>(src1 + width - A, a[1]);
                    LoadTail3<false, 1>(src2 + width - A, a[2]);
                    ContourMetrics<false, true>(a, _dst);
                    _dst.Flush();
               }

                dst += dstStride;
            }
        }

        void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ContourMetrics<true>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
            else
                ContourMetrics<false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }
    }
#endif// SIMD_VSX_ENABLE
}
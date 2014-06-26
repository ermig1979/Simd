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
#include "Simd/SimdConversion.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        template <bool align, bool first>
        SIMD_INLINE void YuvToBgr(const v128_u8 & y, const v128_u8 & u, const v128_u8 & v, Storer<align> & bgr)
        {
            const v128_u8 blue = YuvToBlue(y, u);
            const v128_u8 green = YuvToGreen(y, u, v);
            const v128_u8 red = YuvToRed(y, v);

            Store<align, first>(bgr, InterleaveBgr<0>(blue, green, red));
            Store<align, false>(bgr, InterleaveBgr<1>(blue, green, red));
            Store<align, false>(bgr, InterleaveBgr<2>(blue, green, red));
        }

        template <bool align, bool first> 
        SIMD_INLINE void Yuv420pToBgr(const uint8_t * y, const v128_u8 & u, const v128_u8 & v, Storer<align> & bgr)
        {
            YuvToBgr<align, first>(Load<align>(y + 0), (v128_u8)UnpackLoU8(u, u), (v128_u8)UnpackLoU8(v, v), bgr);
            YuvToBgr<align, false>(Load<align>(y + A), (v128_u8)UnpackHiU8(u, u), (v128_u8)UnpackHiU8(v, v), bgr);
        }

        template <bool align> void Yuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, 
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert((width%2 == 0) && (height%2 == 0) && (width >= DA) && (height >= 2));
            if(align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) &&  Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for(size_t row = 0; row < height; row += 2)
            {
                Storer<align> _bgr0(bgr), _bgr1(bgr + bgrStride);
                v128_u8 _u = Load<align>(u);
                v128_u8 _v = Load<align>(v);
                Yuv420pToBgr<align, true>(y, _u, _v, _bgr0);
                Yuv420pToBgr<align, true>(y + yStride, _u, _v, _bgr1);
                for(size_t colUV = A, colY = DA; colY < bodyWidth; colY += DA, colUV += A)
                {
                    v128_u8 _u = Load<align>(u + colUV);
                    v128_u8 _v = Load<align>(v + colUV);
                    Yuv420pToBgr<align, false>(y + colY, _u, _v, _bgr0);
                    Yuv420pToBgr<align, false>(y + colY + yStride, _u, _v, _bgr1);
                }
                _bgr0.Flush();
                _bgr1.Flush();

                if(tail)
                {
                   size_t offset = width - DA;
                    Storer<false> _bgr0(bgr + 3*offset), _bgr1(bgr + 3*offset + bgrStride);
                    v128_u8 _u = Load<false>(u + offset/2);
                    v128_u8 _v = Load<false>(v + offset/2);
                    Yuv420pToBgr<false, true>(y + offset, _u, _v, _bgr0);
                    Yuv420pToBgr<false, true>(y + offset + yStride, _u, _v, _bgr1);
                    _bgr0.Flush();
                    _bgr1.Flush();
                }
                y += 2*yStride;
                u += uStride;
                v += vStride;
                bgr += 2*bgrStride;
            }
        }

        void Yuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, 
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) 
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv420pToBgr<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
            else
                Yuv420pToBgr<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
        }
    }
#endif// SIMD_VSX_ENABLE
}
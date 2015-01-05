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
#include "Simd/SimdBase.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width, size_t height)
                {
                    _p = Allocate(sizeof(int)*(2*height + width) + sizeof(uint16_t)*4*width);
                    ix = (int*)_p;
                    ax = (uint16_t*)(ix + width);
                    pbx[0] = (uint16_t*)(ax + 2*width);
                    pbx[1] = pbx[0] + width;
                    iy = (int*)(pbx[1] + width);
                    ay = iy + height;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                int * ix;
                uint16_t * ax;
                int * iy;
                int * ay;
                uint16_t * pbx[2];
            private:
                void *_p;
            };
        }

        void EstimateAlphaIndexGrayX(size_t srcSize, size_t dstSize, int * indexes, uint16_t * alphas)
        {
            float scale = (float)srcSize/dstSize;

            for(size_t i = 0; i < dstSize; ++i)
            {
                float alpha = (float)((i + 0.5)*scale - 0.5);
                ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                alpha -= index;

                if(index < 0)
                {
                    index = 0;
                    alpha = 0;
                }

                if(index > (ptrdiff_t)srcSize - 2)
                {
                    index = srcSize - 2;
                    alpha = 1;
                }

                indexes[i] = (int)index;
                alphas[1] = (uint16_t)(alpha * Base::FRACTION_RANGE + 0.5);
                alphas[0] = (uint16_t)(Base::FRACTION_RANGE - alphas[1]);
                alphas += 2;
            }
        }

        SIMD_INLINE void InterpolateGrayX(const uint16_t * src, const uint16_t * alpha, uint16_t * dst)
        {
            v128_u8 _src = (v128_u8)Load<true>(src);
            v128_u32 lo = vec_msum(UnpackLoU8(_src), Load<true>(alpha), K32_00000000);
            v128_u32 hi = vec_msum(UnpackHiU8(_src), Load<true>(alpha + HA), K32_00000000);
            Store<true>(dst, vec_pack(lo, hi));
        }

        const v128_u16 K16_FRACTION_ROUND_TERM = SIMD_VEC_SET1_EPI16(Base::BILINEAR_ROUND_TERM);
        const v128_u16 K16_BILINEAR_SHIFT = SIMD_VEC_SET1_EPI16(Base::BILINEAR_SHIFT);

        template<bool align> SIMD_INLINE v128_u8 InterpolateGrayY(const Buffer & buffer, size_t offset, v128_u16 a[2])
        {
            v128_u16 lo = vec_sr(vec_mladd(Load<align>(buffer.pbx[0] + offset), a[0], vec_mladd(Load<align>(buffer.pbx[1] + offset), a[1], K16_FRACTION_ROUND_TERM)), K16_BILINEAR_SHIFT);
            offset += HA;
            v128_u16 hi = vec_sr(vec_mladd(Load<align>(buffer.pbx[0] + offset), a[0], vec_mladd(Load<align>(buffer.pbx[1] + offset), a[1], K16_FRACTION_ROUND_TERM)), K16_BILINEAR_SHIFT);
            return vec_pack(lo, hi);
        }

        template<bool align> void ResizeBilinearGray(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(dstWidth >= A);
            if(align)
                assert(Aligned(dst) && Aligned(dstStride));

            size_t bufferWidth = AlignHi(dstWidth, HA);
            size_t alignedWidth = AlignLo(dstWidth, A);

            Buffer buffer(bufferWidth, dstHeight);

            Base::EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);

            EstimateAlphaIndexGrayX(srcWidth, dstWidth, buffer.ix, buffer.ax);

            ptrdiff_t previous = -2;

            v128_u16 s[2], a[2];

            for(size_t yDst = 0; yDst < dstHeight; yDst++, dst += dstStride)
            {
                a[0] = SetU16(Base::FRACTION_RANGE - buffer.ay[yDst]);
                a[1] = SetU16(buffer.ay[yDst]);

                ptrdiff_t sy = buffer.iy[yDst];
                int k = 0;

                if(sy == previous)
                    k = 2;
                else if(sy == previous + 1)
                {
                    Swap(buffer.pbx[0], buffer.pbx[1]);
                    k = 1;
                }

                previous = sy;

                for(; k < 2; k++)
                {
                    uint16_t * pb = buffer.pbx[k];
                    const uint8_t * ps = src + (sy + k)*srcStride;
                    for(size_t x = 0; x < dstWidth; x++)
                        pb[x] = *(uint16_t*)(ps + buffer.ix[x]);

                    for(size_t i = 0; i < bufferWidth; i += HA)
                    {
                        InterpolateGrayX(pb + i, buffer.ax + 2*i, pb + i);
                    }
                }

                Storer<align> _dst(dst);
                Store<align, true>(_dst, InterpolateGrayY<true>(buffer, 0, a));
                for(size_t col = A; col < alignedWidth; col += A)
                    Store<align, false>(_dst, InterpolateGrayY<true>(buffer, col, a));
                Flush(_dst);
                if(dstWidth != alignedWidth)
                    Store<false>(dst + dstWidth - A, InterpolateGrayY<false>(buffer, dstWidth - A, a));
            }
        }

        void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            if(channelCount == 1)
            {
                if(Aligned(dst) && Aligned(dstStride))
                    ResizeBilinearGray<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
                else
                    ResizeBilinearGray<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            }
            else
                Base::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
        }	
    }
#endif// SIMD_VSX_ENABLE
}
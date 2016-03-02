/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t size, size_t width, size_t height)
                {
                    _p = Allocate(3 * size + sizeof(int)*(2 * height + width));
                    bx[0] = (uint8_t*)_p;
                    bx[1] = bx[0] + size;
                    ax = bx[1] + size;
                    ix = (int*)(ax + size);
                    iy = ix + width;
                    ay = iy + height;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint8_t * bx[2];
                uint8_t * ax;
                int * ix;
                int * ay;
                int * iy;
            private:
                void *_p;
            };
        }

        template <size_t channelCount> void EstimateAlphaIndexX(size_t srcSize, size_t dstSize, int * indexes, uint8_t * alphas)
        {
            float scale = (float)srcSize / dstSize;

            for (size_t i = 0; i < dstSize; ++i)
            {
                float alpha = (float)((i + 0.5)*scale - 0.5);
                ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                alpha -= index;

                if (index < 0)
                {
                    index = 0;
                    alpha = 0;
                }

                if (index >(ptrdiff_t)srcSize - 2)
                {
                    index = srcSize - 2;
                    alpha = 1;
                }

                indexes[i] = (int)index;
                alphas[1] = (uint8_t)(alpha * Base::FRACTION_RANGE + 0.5);
                alphas[0] = (uint8_t)(Base::FRACTION_RANGE - alphas[1]);
                for (size_t channel = 1; channel < channelCount; channel++)
                    ((uint16_t*)alphas)[channel] = *(uint16_t*)alphas;
                alphas += 2 * channelCount;
            }
        }

        template <size_t channelCount> void InterpolateX(const uint8_t * alpha, uint8_t * buffer);

        template <> SIMD_INLINE void InterpolateX<1>(const uint8_t * alpha, uint8_t * buffer)
        {
            uint8x8x2_t a = vld2_u8(alpha);
            uint8x8x2_t b = vld2_u8(buffer);
            Store<true>(buffer, (uint8x16_t)vaddq_u16(vmull_u8(a.val[0], b.val[0]), vmull_u8(a.val[1], b.val[1])));
        }

        SIMD_INLINE void InterpolateX2(const uint8_t * alpha, uint8_t * buffer)
        {
            uint8x8x2_t a = vld2_u8(alpha);
            uint16x4x2_t b = vld2_u16((uint16_t*)buffer);
            Store<true>(buffer, (uint8x16_t)vaddq_u16(vmull_u8(a.val[0], (uint8x8_t)b.val[0]), vmull_u8(a.val[1], (uint8x8_t)b.val[1])));
        }

        template <> SIMD_INLINE void InterpolateX<2>(const uint8_t * alpha, uint8_t * buffer)
        {
            InterpolateX2(alpha + 0, buffer + 0);
            InterpolateX2(alpha + A, buffer + A);
        }

        SIMD_INLINE void InterpolateX3(const uint8_t * alpha, const uint8_t * src, uint8_t * dst)
        {
            uint8x8x2_t a = vld2_u8(alpha);
            uint8x8x2_t b = vld2_u8(src);
            Store<true>(dst, (uint8x16_t)vaddq_u16(vmull_u8(a.val[0], b.val[0]), vmull_u8(a.val[1], b.val[1])));
        }

        template <> SIMD_INLINE void InterpolateX<3>(const uint8_t * alpha, uint8_t * buffer)
        {
            uint8_t b[3 * A];
            uint8x16x3_t _b = vld3q_u8(buffer);
            vst3q_u16((uint16_t*)b, *(uint16x8x3_t*)&_b);
            InterpolateX3(alpha + 0 * A, b + 0 * A, buffer + 0 * A);
            InterpolateX3(alpha + 1 * A, b + 1 * A, buffer + 1 * A);
            InterpolateX3(alpha + 2 * A, b + 2 * A, buffer + 2 * A);
        }        

        SIMD_INLINE void InterpolateX4(const uint8_t * alpha, uint8_t * buffer)
        {
            uint8x8x2_t a = vld2_u8(alpha);
            uint32x2x2_t b = vld2_u32((uint32_t*)buffer);
            Store<true>(buffer, (uint8x16_t)vaddq_u16(vmull_u8(a.val[0], (uint8x8_t)b.val[0]), vmull_u8(a.val[1], (uint8x8_t)b.val[1])));
        }

        template <> SIMD_INLINE void InterpolateX<4>(const uint8_t * alpha, uint8_t * buffer)
        {
            InterpolateX4(alpha + 0*A, buffer + 0*A);
            InterpolateX4(alpha + 1*A, buffer + 1*A);
            InterpolateX4(alpha + 2*A, buffer + 2*A);
            InterpolateX4(alpha + 3*A, buffer + 3*A);
        }

        const uint16x8_t K16_FRACTION_ROUND_TERM = SIMD_VEC_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE uint16x8_t InterpolateY(const uint16_t * pbx0, const uint16_t * pbx1, uint16x8_t alpha[2])
        {
            uint16x8_t sum = vaddq_u16(vmulq_u16(Load<align>(pbx0), alpha[0]), vmulq_u16(Load<align>(pbx1), alpha[1]));
            return vshrq_n_u16(vaddq_u16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void InterpolateY(const uint8_t * bx0, const uint8_t * bx1, uint16x8_t alpha[2], uint8_t * dst)
        {
            uint16x8_t lo = InterpolateY<align>((uint16_t*)(bx0 + 0), (uint16_t*)(bx1 + 0), alpha); 
            uint16x8_t hi = InterpolateY<align>((uint16_t*)(bx0 + A), (uint16_t*)(bx1 + A), alpha); 
            Store<false>(dst, PackU16(lo, hi));
        }

        template <size_t channelCount> void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(dstWidth >= A);

            struct One { uint8_t channels[channelCount]; };
            struct Two { uint8_t channels[channelCount*2]; };

            size_t size = 2*dstWidth*channelCount;
            size_t bufferSize = AlignHi(dstWidth, A)*channelCount*2;
            size_t alignedSize = AlignHi(size, DA) - DA;
            const size_t step = A*channelCount;

            Buffer buffer(bufferSize, dstWidth, dstHeight);

            Base::EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);

            EstimateAlphaIndexX<channelCount>(srcWidth, dstWidth, buffer.ix, buffer.ax);

            ptrdiff_t previous = -2;

            uint16x8_t a[2];

            for(size_t yDst = 0; yDst < dstHeight; yDst++, dst += dstStride)
            {
                a[0] = vdupq_n_u16(Base::FRACTION_RANGE - buffer.ay[yDst]);
                a[1] = vdupq_n_u16(buffer.ay[yDst]);

                ptrdiff_t sy = buffer.iy[yDst];
                int k = 0;

                if(sy == previous)
                    k = 2;
                else if(sy == previous + 1)
                {
                    Swap(buffer.bx[0], buffer.bx[1]);
                    k = 1;
                }

                previous = sy;

                for(; k < 2; k++)
                {
                    Two * pb = (Two *)buffer.bx[k];
                    const One * psrc = (const One *)(src + (sy + k)*srcStride);
                    for(size_t x = 0; x < dstWidth; x++)
                        pb[x] = *(Two *)(psrc + buffer.ix[x]);

                    uint8_t * pbx = buffer.bx[k];
                    for(size_t i = 0; i < bufferSize; i += step)
                        InterpolateX<channelCount>(buffer.ax + i, pbx + i);
                }

                for(size_t ib = 0, id = 0; ib < alignedSize; ib += DA, id += A)
                    InterpolateY<true>(buffer.bx[0] + ib, buffer.bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                InterpolateY<false>(buffer.bx[0] + i, buffer.bx[1] + i, a, dst + i/2);
            }
        }

        void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            switch(channelCount)
            {
            case 1: 
                ResizeBilinear<1>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride); 
                break;
            case 2: 
                ResizeBilinear<2>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride); 
                break;
            case 3: 
                ResizeBilinear<3>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride); 
                break;
            case 4: 
                ResizeBilinear<4>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride); 
                break;
            default: 
                Base::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount); 
            }
        }	
    }
#endif
}


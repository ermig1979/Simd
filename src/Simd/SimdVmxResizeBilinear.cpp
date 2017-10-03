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
#include "Simd/SimdBase.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
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

                if (index > (ptrdiff_t)srcSize - 2)
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
            v128_u8 _alpha = Load<true>(alpha);
            v128_u8 _buffer = Load<true>(buffer);
            Store<true>(buffer, (v128_u8)vec_add(vec_mule(_alpha, _buffer), vec_mulo(_alpha, _buffer)));
        }

        const v128_u8 K8_PERM_X2 = SIMD_VEC_SETR_EPI8(0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

        SIMD_INLINE void InterpolateX2(const uint8_t * alpha, uint8_t * buffer)
        {
            v128_u8 _alpha = Load<true>(alpha);
            v128_u8 _buffer = vec_perm(Load<true>(buffer), K8_00, K8_PERM_X2);
            Store<true>(buffer, (v128_u8)vec_add(vec_mule(_alpha, _buffer), vec_mulo(_alpha, _buffer)));
        }

        template <> SIMD_INLINE void InterpolateX<2>(const uint8_t * alpha, uint8_t * buffer)
        {
            InterpolateX2(alpha + 0, buffer + 0);
            InterpolateX2(alpha + A, buffer + A);
        }

        const v128_u8 K8_PERM_X3_00 = SIMD_VEC_SETR_EPI8(0x00, 0x03, 0x01, 0x04, 0x02, 0x05, 0x06, 0x09, 0x07, 0x0A, 0x08, 0x0B, 0x0C, 0x0F, 0x0D, 0x10);
        const v128_u8 K8_PERM_X3_10 = SIMD_VEC_SETR_EPI8(0x0E, 0x11, 0x12, 0x15, 0x13, 0x16, 0x14, 0x17, 0x18, 0x1B, 0x19, 0x1C, 0x1A, 0x1D, 0x1E, 0x00);
        const v128_u8 K8_PERM_X3_11 = SIMD_VEC_SETR_EPI8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x11);
        const v128_u8 K8_PERM_X3_20 = SIMD_VEC_SETR_EPI8(0x0F, 0x12, 0x10, 0x13, 0x14, 0x17, 0x15, 0x18, 0x16, 0x19, 0x1A, 0x1D, 0x1B, 0x1E, 0x1C, 0x1F);

        template <> SIMD_INLINE void InterpolateX<3>(const uint8_t * alpha, uint8_t * buffer)
        {
            v128_u8 buffer0 = Load<true>(buffer + 0 * A);
            v128_u8 buffer1 = Load<true>(buffer + 1 * A);
            v128_u8 buffer2 = Load<true>(buffer + 2 * A);
            v128_u8 value0 = vec_perm(buffer0, buffer1, K8_PERM_X3_00);
            v128_u8 alpha0 = Load<true>(alpha + 0 * A);
            Store<true>(buffer + 0 * A, (v128_u8)vec_add(vec_mule(alpha0, value0), vec_mulo(alpha0, value0)));
            v128_u8 value1 = vec_perm(vec_perm(buffer0, buffer1, K8_PERM_X3_10), buffer2, K8_PERM_X3_11);
            v128_u8 alpha1 = Load<true>(alpha + 1 * A);
            Store<true>(buffer + 1 * A, (v128_u8)vec_add(vec_mule(alpha1, value1), vec_mulo(alpha1, value1)));
            v128_u8 value2 = vec_perm(buffer1, buffer2, K8_PERM_X3_20);
            v128_u8 alpha2 = Load<true>(alpha + 2 * A);
            Store<true>(buffer + 2 * A, (v128_u8)vec_add(vec_mule(alpha2, value2), vec_mulo(alpha2, value2)));
        }

        const v128_u8 K8_PERM_X4 = SIMD_VEC_SETR_EPI8(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

        SIMD_INLINE void InterpolateX4(const uint8_t * alpha, uint8_t * buffer)
        {
            v128_u8 _alpha = Load<true>(alpha);
            v128_u8 _buffer = vec_perm(Load<true>(buffer), K8_00, K8_PERM_X4);
            Store<true>(buffer, (v128_u8)vec_add(vec_mule(_alpha, _buffer), vec_mulo(_alpha, _buffer)));
        }

        template <> SIMD_INLINE void InterpolateX<4>(const uint8_t * alpha, uint8_t * buffer)
        {
            InterpolateX4(alpha + 0 * A, buffer + 0 * A);
            InterpolateX4(alpha + 1 * A, buffer + 1 * A);
            InterpolateX4(alpha + 2 * A, buffer + 2 * A);
            InterpolateX4(alpha + 3 * A, buffer + 3 * A);
        }

        const v128_u16 K16_FRACTION_ROUND_TERM = SIMD_VEC_SET1_EPI16(Base::BILINEAR_ROUND_TERM);
        const v128_u16 K16_BILINEAR_SHIFT = SIMD_VEC_SET1_EPI16(Base::BILINEAR_SHIFT);

        template<bool align> SIMD_INLINE v128_u8 InterpolateY(const uint8_t * bx0, const uint8_t * bx1, v128_u16 a[2])
        {
            v128_u16 lo = vec_sr(vec_mladd(Load<align>((uint16_t*)(bx0 + 0)), a[0], vec_mladd(Load<align>((uint16_t*)(bx1 + 0)), a[1], K16_FRACTION_ROUND_TERM)), K16_BILINEAR_SHIFT);
            v128_u16 hi = vec_sr(vec_mladd(Load<align>((uint16_t*)(bx0 + A)), a[0], vec_mladd(Load<align>((uint16_t*)(bx1 + A)), a[1], K16_FRACTION_ROUND_TERM)), K16_BILINEAR_SHIFT);
            return vec_pack(lo, hi);
        }

        template <size_t channelCount, bool align> void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(dstWidth >= A);

            struct One { uint8_t channels[channelCount]; };
            struct Two { uint8_t channels[channelCount * 2]; };

            size_t size = 2 * dstWidth*channelCount;
            size_t bufferSize = AlignHi(dstWidth, A)*channelCount * 2;
            size_t alignedSize = AlignHi(size, DA) - DA;
            const size_t step = A*channelCount;

            Buffer buffer(bufferSize, dstWidth, dstHeight);

            Base::EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);

            EstimateAlphaIndexX<channelCount>(srcWidth, dstWidth, buffer.ix, buffer.ax);

            ptrdiff_t previous = -2;

            v128_u16 a[2];

            for (size_t yDst = 0; yDst < dstHeight; yDst++, dst += dstStride)
            {
                a[0] = SetU16(int16_t(Base::FRACTION_RANGE - buffer.ay[yDst]));
                a[1] = SetU16(int16_t(buffer.ay[yDst]));

                ptrdiff_t sy = buffer.iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(buffer.bx[0], buffer.bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    Two * pb = (Two *)buffer.bx[k];
                    const One * psrc = (const One *)(src + (sy + k)*srcStride);
                    for (size_t x = 0; x < dstWidth; x++)
                        pb[x] = *(Two *)(psrc + buffer.ix[x]);

                    uint8_t * pbx = buffer.bx[k];
                    for (size_t i = 0; i < bufferSize; i += step)
                        InterpolateX<channelCount>(buffer.ax + i, pbx + i);
                }

                Storer<align> _dst(dst);
                Store<align, true>(_dst, InterpolateY<true>(buffer.bx[0], buffer.bx[1], a));
                for (size_t i = DA; i < alignedSize; i += DA)
                    Store<align, false>(_dst, InterpolateY<true>(buffer.bx[0] + i, buffer.bx[1] + i, a));
                Flush(_dst);
                size_t i = size - DA;
                Store<false>(dst + i / 2, InterpolateY<false>(buffer.bx[0] + i, buffer.bx[1] + i, a));
            }
        }

        template <size_t channelCount> void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if (Aligned(dst) && Aligned(dstStride))
                ResizeBilinear<channelCount, true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ResizeBilinear<channelCount, false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }

        void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            switch (channelCount)
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
#endif// SIMD_VMX_ENABLE
}

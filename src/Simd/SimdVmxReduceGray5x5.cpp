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
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t)*(5 * width + A));
                    in0 = (uint16_t*)_p;
                    in1 = in0 + width;
                    out0 = in1 + width;
                    out1 = out0 + width;
                    dst = out1 + width + HA;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * in0;
                uint16_t * in1;
                uint16_t * out0;
                uint16_t * out1;
                uint16_t * dst;
            private:
                void *_p;
            };
        }

        template <bool compensation> SIMD_INLINE v128_u16 DivideBy256(v128_u16 value);

        template <> SIMD_INLINE v128_u16 DivideBy256<true>(v128_u16 value)
        {
            return vec_sr(vec_add(value, K16_0080), K16_0008);
        }

        template <> SIMD_INLINE v128_u16 DivideBy256<false>(v128_u16 value)
        {
            return vec_sr(value, K16_0008);
        }

        template <bool align> SIMD_INLINE void LoadUnpacked(const uint8_t * src, v128_u16 & lo, v128_u16 & hi)
        {
            v128_u8 t = Load<align>(src);
            lo = UnpackLoU8(t);
            hi = UnpackHiU8(t);
        }

        template<bool align> SIMD_INLINE void FirstRow5x5(v128_u16 src, Buffer & buffer, size_t offset)
        {
            Store<align>(buffer.in0 + offset, src);
            Store<align>(buffer.in1 + offset, vec_mladd(src, K16_0005, K16_0000));
        }

        template<bool srcAlign, bool dstAlign> SIMD_INLINE void FirstRow5x5(const uint8_t * src, Buffer & buffer, size_t offset)
        {
            v128_u16 lo, hi;
            LoadUnpacked<srcAlign>(src + offset, lo, hi);
            FirstRow5x5<dstAlign>(lo, buffer, offset);
            FirstRow5x5<dstAlign>(hi, buffer, offset + HA);
        }

        template<bool align> SIMD_INLINE void MainRowY5x5(v128_u16 odd, v128_u16 even, Buffer & buffer, size_t offset)
        {
            v128_u16 cp = vec_mladd(odd, K16_0004, K16_0000);
            v128_u16 c0 = Load<align>(buffer.in0 + offset);
            v128_u16 c1 = Load<align>(buffer.in1 + offset);
            Store<align>(buffer.dst + offset, vec_add(even, vec_add(c1, vec_mladd(c0, K16_0006, cp))));
            Store<align>(buffer.out1 + offset, vec_add(c0, cp));
            Store<align>(buffer.out0 + offset, even);
        }

        template<bool srcAlign, bool dstAlign> SIMD_INLINE void MainRowY5x5(const uint8_t * odd, const uint8_t * even, Buffer & buffer, size_t offset)
        {
            v128_u16 evenLo, evenHi, oddLo, oddHi;
            LoadUnpacked<srcAlign>(even + offset, evenLo, evenHi);
            LoadUnpacked<srcAlign>(odd + offset, oddLo, oddHi);
            MainRowY5x5<dstAlign>(oddLo, evenLo, buffer, offset);
            MainRowY5x5<dstAlign>(oddHi, evenHi, buffer, offset + HA);
        }

        template <bool align, bool compensation> SIMD_INLINE v128_u16 MainRowX5x5(uint16_t * dst)
        {
            v128_u16 t0 = Load<false>(dst - 2);
            v128_u16 t1 = Load<false>(dst - 1);
            v128_u16 t2 = Load<align>(dst);
            v128_u16 t3 = Load<false>(dst + 1);
            v128_u16 t4 = Load<false>(dst + 2);
            t2 = vec_mladd(t2, K16_0006, vec_mladd(vec_add(t1, t3), K16_0004, vec_add(t0, t4)));
            return DivideBy256<compensation>(t2);
        }

        template <bool align, bool compensation> SIMD_INLINE v128_u16 MainRowX5x5_16(Buffer & buffer, size_t offset)
        {
            const v128_u16 lo = MainRowX5x5<align, compensation>(buffer.dst + offset);
            const v128_u16 hi = MainRowX5x5<align, compensation>(buffer.dst + offset + HA);
            return vec_mule((v128_u8)vec_packsu(lo, hi), K8_01);
        }

        template <bool align, bool compensation> SIMD_INLINE v128_u8 MainRowX5x5(Buffer & buffer, size_t offset)
        {
            v128_u16 lo = MainRowX5x5_16<align, compensation>(buffer, offset);
            v128_u16 hi = MainRowX5x5_16<align, compensation>(buffer, offset + A);
            return vec_packsu(lo, hi);
        }

        template <bool align, bool compensation> void ReduceGray5x5(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= DA);
            if (align)
                assert(Aligned(src) && Aligned(srcStride));

            size_t alignedWidth = Simd::AlignLo(srcWidth, DA);
            size_t bufferDstTail = Simd::AlignHi(srcWidth - DA, 2);

            Buffer buffer(Simd::AlignHi(srcWidth, A));

            for (size_t col = 0; col < alignedWidth; col += A)
                FirstRow5x5<align, true>(src, buffer, col);
            if (alignedWidth != srcWidth)
            {
                FirstRow5x5<false, false>(src, buffer, srcWidth - DA);
                FirstRow5x5<false, false>(src, buffer, srcWidth - A);
            }
            src += srcStride;

            for (size_t row = 1; row <= srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t * odd = src - (row < srcHeight ? 0 : srcStride);
                const uint8_t * even = odd + (row < srcHeight - 1 ? srcStride : 0);

                for (size_t col = 0; col < alignedWidth; col += A)
                    MainRowY5x5<align, true>(odd, even, buffer, col);
                if (alignedWidth != srcWidth)
                {
                    MainRowY5x5<false, false>(odd, even, buffer, srcWidth - DA);
                    MainRowY5x5<false, false>(odd, even, buffer, srcWidth - A);
                }

                Swap(buffer.in0, buffer.out0);
                Swap(buffer.in1, buffer.out1);

                buffer.dst[-2] = buffer.dst[0];
                buffer.dst[-1] = buffer.dst[0];
                buffer.dst[srcWidth] = buffer.dst[srcWidth - 1];
                buffer.dst[srcWidth + 1] = buffer.dst[srcWidth - 1];

                Storer<false> _dst(dst);
                Store<false, true>(_dst, MainRowX5x5<true, compensation>(buffer, 0));
                for (size_t srcCol = DA; srcCol < alignedWidth; srcCol += DA)
                    Store<false, false>(_dst, MainRowX5x5<true, compensation>(buffer, srcCol));
                Flush(_dst);
                if (alignedWidth != srcWidth)
                    Store<false>(dst + dstWidth - A, MainRowX5x5<false, compensation>(buffer, bufferDstTail));
            }
        }

        template <bool compensation> void ReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride))
                ReduceGray5x5<true, compensation>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray5x5<false, compensation>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }

        void ReduceGray5x5(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray5x5<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray5x5<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}

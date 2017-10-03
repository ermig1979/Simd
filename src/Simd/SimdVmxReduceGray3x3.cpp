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
        template <bool compensation> SIMD_INLINE v128_u16 DivideBy16(v128_u16 value);

        template <> SIMD_INLINE v128_u16 DivideBy16<true>(v128_u16 value)
        {
            return vec_sr(vec_add(value, K16_0008), K16_0004);
        }

        template <> SIMD_INLINE v128_u16 DivideBy16<false>(v128_u16 value)
        {
            return vec_sr(value, K16_0004);
        }

        template<bool align> SIMD_INLINE v128_u16 ReduceColNose(const uint8_t * p)
        {
            const v128_u8 t = Load<align>(p);
            return BinomialSum(vec_mule(LoadBeforeFirst<1>(t), K8_01), vec_mule(t, K8_01), vec_mulo(t, K8_01));
        }

        template<bool align> SIMD_INLINE void ReduceColNose(const uint8_t * s[3], v128_u16 a[3])
        {
            a[0] = ReduceColNose<align>(s[0]);
            a[1] = ReduceColNose<align>(s[1]);
            a[2] = ReduceColNose<align>(s[2]);
        }

        template<bool align> SIMD_INLINE v128_u16 ReduceColBody(const uint8_t * p)
        {
            const v128_u8 t0 = Load<false>(p - 1);
            const v128_u8 t1 = Load<align>(p);
            return BinomialSum(vec_mule(t0, K8_01), vec_mule(t1, K8_01), vec_mulo(t1, K8_01));
        }

        template<bool align> SIMD_INLINE void ReduceColBody(const uint8_t * s[3], size_t offset, v128_u16 a[3])
        {
            a[0] = ReduceColBody<align>(s[0] + offset);
            a[1] = ReduceColBody<align>(s[1] + offset);
            a[2] = ReduceColBody<align>(s[2] + offset);
        }

        template <bool compensation> SIMD_INLINE v128_u8 ReduceRow(const v128_u16 lo[3], const v128_u16 hi[3])
        {
            return vec_packsu(
                DivideBy16<compensation>(BinomialSum(lo[0], lo[1], lo[2])),
                DivideBy16<compensation>(BinomialSum(hi[0], hi[1], hi[2])));
        }

        template<bool align, bool compensation> void ReduceGray3x3(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth >= DA && (srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t lastOddCol = srcWidth - AlignLo(srcWidth, 2);
            size_t bodyWidth = AlignLo(srcWidth, DA);
            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t * s[3];
                s[1] = src;
                s[0] = s[1] - (row ? srcStride : 0);
                s[2] = s[1] + (row != srcHeight - 1 ? srcStride : 0);

                Storer<align> _dst(dst);
                v128_u16 lo[3], hi[3];
                ReduceColNose<align>(s, lo);
                ReduceColBody<align>(s, A, hi);
                Store<align, true>(_dst, ReduceRow<compensation>(lo, hi));
                for (size_t srcCol = DA, dstCol = A; srcCol < bodyWidth; srcCol += DA)
                {
                    ReduceColBody<align>(s, srcCol, lo);
                    ReduceColBody<align>(s, srcCol + A, hi);
                    Store<align, false>(_dst, ReduceRow<compensation>(lo, hi));
                }
                Flush(_dst);

                if (bodyWidth != srcWidth)
                {
                    size_t srcCol = srcWidth - DA - lastOddCol;
                    size_t dstCol = dstWidth - A - lastOddCol;
                    ReduceColBody<false>(s, srcCol, lo);
                    ReduceColBody<false>(s, srcCol + A, hi);
                    Store<false>(dst + dstCol, ReduceRow<compensation>(lo, hi));
                    if (lastOddCol)
                        dst[dstWidth - 1] = Base::GaussianBlur3x3<compensation>(s[0] + srcWidth, s[1] + srcWidth, s[2] + srcWidth, -2, -1, -1);
                }
            }
        }

        template<bool align> void ReduceGray3x3(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray3x3<align, true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray3x3<align, false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }

        void ReduceGray3x3(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ReduceGray3x3<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
            else
                ReduceGray3x3<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
        }
    }
#endif// SIMD_VMX_ENABLE
}

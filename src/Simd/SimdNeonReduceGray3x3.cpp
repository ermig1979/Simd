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
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool compensation> SIMD_INLINE uint16x8_t DivideBy16(uint16x8_t value);

        template <> SIMD_INLINE uint16x8_t DivideBy16<true>(uint16x8_t value)
        {
            return vshrq_n_u16(vaddq_u16(value, K16_0008), 4);
        }

        template <> SIMD_INLINE uint16x8_t DivideBy16<false>(uint16x8_t value)
        {
            return vshrq_n_u16(value, 4);
        }

        template<bool align> SIMD_INLINE uint16x8_t ReduceColNose(const uint8_t * src)
        {
            uint8x16_t t12 = Load<align>(src);
            uint8x16_t t01 = LoadBeforeFirst<1>(t12);
            return vaddq_u16(vpaddlq_u8(t01), vpaddlq_u8(t12));
        }

        template<bool align> SIMD_INLINE uint16x8_t ReduceColBody(const uint8_t * src)
        {
            uint8x16_t t01 = Load<false>(src - 1);
            uint8x16_t t12 = Load<align>(src);
            return vaddq_u16(vpaddlq_u8(t01), vpaddlq_u8(t12));
        }

        template <bool compensation> SIMD_INLINE uint8x8_t ReduceRow(const uint16x8_t & r0, const uint16x8_t & r1, const uint16x8_t & r2)
        {
            return vmovn_u16(DivideBy16<compensation>(BinomialSum16(r0, r1, r2)));
        }

        template<bool align, bool compensation> void ReduceGray3x3(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(srcWidth >= A && (srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);
            if (align)
                assert(Aligned(src) && Aligned(srcStride));

            size_t lastOddCol = srcWidth - AlignLo(srcWidth, 2);
            size_t bodyWidth = AlignLo(srcWidth, A);
            for (size_t row = 0; row < srcHeight; row += 2, dst += dstStride, src += 2 * srcStride)
            {
                const uint8_t * s1 = src;
                const uint8_t * s0 = s1 - (row ? srcStride : 0);
                const uint8_t * s2 = s1 + (row != srcHeight - 1 ? srcStride : 0);

                vst1_u8(dst, ReduceRow<compensation>(ReduceColNose<align>(s0),
                    ReduceColNose<align>(s1), ReduceColNose<align>(s2)));

                for (size_t srcCol = A, dstCol = HA; srcCol < bodyWidth; srcCol += A, dstCol += HA)
                    vst1_u8(dst + dstCol, ReduceRow<compensation>(ReduceColBody<align>(s0 + srcCol),
                        ReduceColBody<align>(s1 + srcCol), ReduceColBody<align>(s2 + srcCol)));

                if (bodyWidth != srcWidth)
                {
                    size_t srcCol = srcWidth - A - lastOddCol;
                    size_t dstCol = dstWidth - HA - lastOddCol;
                    vst1_u8(dst + dstCol, ReduceRow<compensation>(ReduceColBody<false>(s0 + srcCol),
                        ReduceColBody<false>(s1 + srcCol), ReduceColBody<false>(s2 + srcCol)));
                    if (lastOddCol)
                        dst[dstWidth - 1] = Base::GaussianBlur3x3<compensation>(s0 + srcWidth, s1 + srcWidth, s2 + srcWidth, -2, -1, -1);
                }
            }
        }

        template<bool align> void ReduceGray3x3(
            const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (compensation)
                ReduceGray3x3<align, true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray3x3<align, false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }

        void ReduceGray3x3(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int compensation)
        {
            if (Aligned(src) && Aligned(srcStride))
                ReduceGray3x3<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
            else
                ReduceGray3x3<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, compensation);
        }
    }
#endif// SIMD_NEON_ENABLE
}

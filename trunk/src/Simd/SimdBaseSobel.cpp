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
#include "Simd/SimdBase.h"
#include "Simd/SimdMath.h"

namespace Simd
{
	namespace Base
	{
        template <bool abs> int SobelDx(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x2);

        template <> SIMD_INLINE int SobelDx<false>(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x2)
        {
            return (s0[x2] + 2*s1[x2] + s2[x2]) - (s0[x0] + 2*s1[x0] + s2[x0]);
        }

        template <> SIMD_INLINE int SobelDx<true>(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x2)
        {
            return Simd::Abs(SobelDx<false>(s0, s1, s2, x0, x2));
        }

        template <bool abs> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            const uint8_t *src0, *src1, *src2;

            for(size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if(row == 0)
                    src0 = src1;
                if(row == height - 1)
                    src2 = src1;

                dst[0] = SobelDx<abs>(src0, src1, src2, 0, 1);

                for(size_t col = 1; col < width - 1; ++col)
                    dst[col] = SobelDx<abs>(src0, src1, src2, col - 1, col + 1);

                dst[width - 1] = SobelDx<abs>(src0, src1, src2, width - 2, width - 1);

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            SobelDx<false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }

        void SobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            SobelDx<true>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }

        template <bool abs> SIMD_INLINE int SobelDy(const uint8_t *s0, const uint8_t *s2, size_t x0, size_t x1, size_t x2);

        template <> SIMD_INLINE int SobelDy<false>(const uint8_t *s0, const uint8_t *s2, size_t x0, size_t x1, size_t x2)
        {
            return (s2[x0] + 2*s2[x1] + s2[x2]) - (s0[x0] + 2*s0[x1] + s0[x2]);
        }

        template <> SIMD_INLINE int SobelDy<true>(const uint8_t *s0, const uint8_t *s2, size_t x0, size_t x1, size_t x2)
        {
            return Simd::Abs(SobelDy<false>(s0, s2, x0, x1, x2));
        }

        template <bool abs> void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            const uint8_t *src0, *src1, *src2;

            for(size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if(row == 0)
                    src0 = src1;
                if(row == height - 1)
                    src2 = src1;

                dst[0] = SobelDy<abs>(src0, src2, 0, 0, 1);

                for(size_t col = 1; col < width - 1; ++col)
                    dst[col] = SobelDy<abs>(src0, src2, col - 1, col, col + 1);

                dst[width - 1] = SobelDy<abs>(src0, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
            }
        }

        void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            SobelDy<false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }

        void SobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            SobelDy<true>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }
    }
}
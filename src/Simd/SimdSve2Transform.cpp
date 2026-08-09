/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdTransform.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        template<size_t N> SIMD_INLINE void TransformImageReverse(const uint8_t* src, size_t width, uint8_t* dst)
        {
            for (size_t col = 0; col < width; ++col)
                Base::CopyPixel<N>(src + col * N, dst + (width - col - 1) * N);
        }

        template<> SIMD_INLINE void TransformImageReverse<1>(const uint8_t* src, size_t width, uint8_t* dst)
        {
            const size_t A = svcntb();
            const size_t widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            size_t col = 0;
            for (; col < widthA; col += A)
                svst1_u8(body, dst + width - col - A, svrev_u8(svld1_u8(body, src + col)));
            for (; col < width; ++col)
                dst[width - col - 1] = src[col];
        }

        template<> SIMD_INLINE void TransformImageReverse<2>(const uint8_t* src, size_t width, uint8_t* dst)
        {
            const size_t A = svcntb();
            const size_t P = A / 2;
            const size_t widthP = AlignLo(width, P);
            const svbool_t body = svptrue_b8();
            size_t col = 0;
            for (; col < widthP; col += P)
            {
                svuint8_t src0 = svld1_u8(body, src + col * 2);
                svuint8_t dst0 = svreinterpret_u8_u16(svrev_u16(svreinterpret_u16_u8(src0)));
                svst1_u8(body, dst + (width - col - P) * 2, dst0);
            }
            for (; col < width; ++col)
                Base::CopyPixel<2>(src + col * 2, dst + (width - col - 1) * 2);
        }

        template<> SIMD_INLINE void TransformImageReverse<4>(const uint8_t* src, size_t width, uint8_t* dst)
        {
            const size_t A = svcntb();
            const size_t P = A / 4;
            const size_t widthP = AlignLo(width, P);
            const svbool_t body = svptrue_b8();
            size_t col = 0;
            for (; col < widthP; col += P)
            {
                svuint8_t src0 = svld1_u8(body, src + col * 4);
                svuint8_t dst0 = svreinterpret_u8_u32(svrev_u32(svreinterpret_u32_u8(src0)));
                svst1_u8(body, dst + (width - col - P) * 4, dst0);
            }
            for (; col < width; ++col)
                Base::CopyPixel<4>(src + col * 4, dst + (width - col - 1) * 4);
        }

        //-------------------------------------------------------------------------------------------------

        template<size_t N> void TransformImageRotate180(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (height - 1) * dstStride;
            for (size_t row = 0; row < height; ++row)
            {
                TransformImageReverse<N>(src, width, dst);
                src += srcStride;
                dst -= dstStride;
            }
        }

        template<size_t N> void TransformImageTransposeRotate90(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                TransformImageReverse<N>(src, width, dst);
                src += srcStride;
                dst += dstStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<size_t N> void Init(Base::ImageTransforms::TransformPtr transforms[8])
        {
            transforms[SimdTransformRotate180] = TransformImageRotate180<N>;
            transforms[SimdTransformTransposeRotate90] = TransformImageTransposeRotate90<N>;
        }

        struct ImageTransforms :
#ifdef SIMD_NEON_ENABLE
            public Neon::ImageTransforms
#else
            public Base::ImageTransforms
#endif
        {
            ImageTransforms()
#ifdef SIMD_NEON_ENABLE
                : Neon::ImageTransforms::ImageTransforms()
#else
                : Base::ImageTransforms::ImageTransforms()
#endif
            {
                Init<1>(transforms[0]);
                Init<2>(transforms[1]);
                Init<4>(transforms[3]);
            }
        };

        //-------------------------------------------------------------------------------------------------

        void TransformImage(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t* dst, size_t dstStride)
        {
            static ImageTransforms transforms = ImageTransforms();

            transforms.TransformImage(src, srcStride, width, height, pixelSize, transform, dst, dstStride);
        }
    }
#endif// SIMD_SVE2_ENABLE
}

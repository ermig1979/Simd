/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdDefs.h"
#include "Simd/SimdTransform.h"

namespace Simd
{
    namespace Base
    {
        template<size_t N> void TransformImageRotate0(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            size_t rowSize = width * N;
            for (size_t row = 0; row < height; ++row)
            {
                memcpy(dst, src, rowSize);
                src += srcStride;
                dst += dstStride;
            }
        }

        template<size_t N> void TransformImageRotate90(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            dst += (width - 1)*dstStride;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst - col * dstStride);
                src += srcStride;
                dst += N;
            }
        }

        template<size_t N> void TransformImageRotate180(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            dst += (height - 1)*dstStride + (width - 1)*N;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst - col * N);
                src += srcStride;
                dst -= dstStride;
            }
        }

        template<size_t N> void TransformImageRotate270(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            dst += (height - 1)*N;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst + col * dstStride);
                src += srcStride;
                dst -= N;
            }
        }

        template<size_t N> void TransformImageTransposeRotate0(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst + col * dstStride);
                src += srcStride;
                dst += N;
            }
        }

        template<size_t N> void TransformImageTransposeRotate90(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            dst += (width - 1)*N;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst - col * N);
                src += srcStride;
                dst += dstStride;
            }
        }

        template<size_t N> void TransformImageTransposeRotate180(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            dst += (width - 1)*dstStride + (height - 1)*N;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst - col * dstStride);
                src += srcStride;
                dst -= N;
            }
        }

        template<size_t N> void TransformImageTransposeRotate270(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            size_t rowSize = width * N;
            dst += (height - 1)*dstStride;
            for (size_t row = 0; row < height; ++row)
            {
                memcpy(dst, src, rowSize);
                src += srcStride;
                dst -= dstStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t N> void Init(ImageTransforms::TransformPtr transforms[8])
        {
            transforms[SimdTransformRotate0] = TransformImageRotate0<N>;
            transforms[SimdTransformRotate90] = TransformImageRotate90<N>;
            transforms[SimdTransformRotate180] = TransformImageRotate180<N>;
            transforms[SimdTransformRotate270] = TransformImageRotate270<N>;
            transforms[SimdTransformTransposeRotate0] = TransformImageTransposeRotate0<N>;
            transforms[SimdTransformTransposeRotate90] = TransformImageTransposeRotate90<N>;
            transforms[SimdTransformTransposeRotate180] = TransformImageTransposeRotate180<N>;
            transforms[SimdTransformTransposeRotate270] = TransformImageTransposeRotate270<N>;
        }

        ImageTransforms::ImageTransforms()
        {
            Init<1>(transforms[0]);
            Init<2>(transforms[1]);
            Init<3>(transforms[2]);
            Init<4>(transforms[3]);
        }

        //-----------------------------------------------------------------------------------------

        void TransformImage(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t * dst, size_t dstStride)
        {
            static ImageTransforms transforms = ImageTransforms();

            transforms.TransformImage(src, srcStride, width, height, pixelSize, transform, dst, dstStride);
        }
    }
}

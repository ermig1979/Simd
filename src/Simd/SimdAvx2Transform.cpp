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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdTransform.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<size_t N> void TransformImageRotate90(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride);

        template<> void TransformImageRotate90<1>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (width - 1) * dstStride;
            size_t width16 = AlignLo(width, 16);
            size_t height8 = AlignLo(height, 8);
            size_t height16 = AlignLo(height, 16);
            size_t row = 0;
            for (; row < height16; row += 16)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Avx2::TransformImageTranspose_1x16x16(src + col * 1, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 16; ++i)
                        Base::CopyPixel<1>(src + col * 1 + i * srcStride, dst - col * dstStride + i * 1);
                src += 16 * srcStride;
                dst += 16;
            }
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Sse41::TransformImageTranspose_1x8x16(src + col * 1, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<1>(src + col * 1 + i * srcStride, dst - col * dstStride + i * 1);
                src += 8 * srcStride;
                dst += 8;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<1>(src + col * 1, dst - col * dstStride);
                src += srcStride;
                dst += 1;
            }
        }

        template<> void TransformImageRotate90<2>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (width - 1) * dstStride;
            size_t width8 = AlignLo(width, 8);
            size_t width16 = AlignLo(width, 16);
            size_t height8 = AlignLo(height, 8);
            size_t height16 = AlignLo(height, 16);
            size_t row = 0;
            for (; row < height16; row += 16)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_2x16x8(src + col * 2, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 16; ++i)
                        Base::CopyPixel<2>(src + col * 2 + i * srcStride, dst - col * dstStride + i * 2);
                src += 16 * srcStride;
                dst += 32;
            }
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Avx2::TransformImageTranspose_2x8x16(src + col * 2, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width8; col += 8)
                    Sse41::TransformImageTranspose_2x8x8(src + col * 2, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<2>(src + col * 2 + i * srcStride, dst - col * dstStride + i * 2);
                src += 8 * srcStride;
                dst += 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<2>(src + col * 2, dst - col * dstStride);
                src += srcStride;
                dst += 2;
            }
        }

        template<> void TransformImageRotate90<3>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (width - 1) * dstStride;
            size_t width4 = AlignLo(width - 5, 4);
            size_t width8 = AlignLo(width - 9, 8);
            size_t height4 = AlignLo(height - 5, 4);
            size_t height8 = AlignLo(height - 9, 4);
            size_t row = 0;
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width4; col += 4)
                    Avx2::TransformImageTranspose_3x8x4(src + col * 3, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<3>(src + col * 3 + i * srcStride, dst - col * dstStride + i * 3);
                src += 8 * srcStride;
                dst += 24;
            }
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_3x4x8(src + col * 3, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width4; col += 4)
                    Sse41::TransformImageTranspose_3x4x4(src + col * 3, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        Base::CopyPixel<3>(src + col * 3 + i * srcStride, dst - col * dstStride + i * 3);
                src += 4 * srcStride;
                dst += 12;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<3>(src + col * 3, dst - col * dstStride);
                src += srcStride;
                dst += 3;
            }
        }

        template<> void TransformImageRotate90<4>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (width - 1) * dstStride;
            size_t width4 = AlignLo(width, 4);
            size_t width8 = AlignLo(width, 8);
            size_t height4 = AlignLo(height, 4);
            size_t height8 = AlignLo(height, 8);
            size_t row = 0;
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_4x8x8(src + col * 4, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width4; col += 4)
                    Avx2::TransformImageTranspose_4x8x4(src + col * 4, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<4>(src + col * 4 + i * srcStride, dst - col * dstStride + i * 4);
                src += 8 * srcStride;
                dst += 32;
            }
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_4x4x8(src + col * 4, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width4; col += 4)
                    Sse41::TransformImageTranspose_4x4x4(src + col * 4, srcStride, dst - col * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        Base::CopyPixel<4>(src + col * 4 + i * srcStride, dst - col * dstStride + i * 4);
                src += 4 * srcStride;
                dst += 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<4>(src + col * 4, dst - col * dstStride);
                src += srcStride;
                dst += 4;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t N> void TransformImageRotate180(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            dst += (height - 1)*dstStride + (width - A)*N;
            size_t widthA = AlignLo(width, A);
            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthDA; col += DA)
                    Avx2::TransformImageMirror64<N>(src + col * N, dst - col * N);
                for (; col < widthA; col += A)
                    Avx2::TransformImageMirror32<N>(src + col * N, dst - col * N);
                if(col < width)
                    Avx2::TransformImageMirror32<N>(src + (width - A) * N, dst - (width - A) * N);
                src += srcStride;
                dst -= dstStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t N> void TransformImageRotate270(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride);

        template<> void TransformImageRotate270<1>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (height - 1) * 1;
            size_t width16 = AlignLo(width, 16);
            size_t height8 = AlignLo(height, 8);
            size_t height16 = AlignLo(height, 16);
            size_t row = 0;
            for (; row < height16; row += 16)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Avx2::TransformImageTranspose_1x16x16(src + col * 1 + 15 * srcStride, -srcStride, dst + col * dstStride - 15, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 16; ++i)
                        Base::CopyPixel<1>(src + col * 1 + i * srcStride, dst + col * dstStride - i * 1);
                src += 16 * srcStride;
                dst -= 16;
            }
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Sse41::TransformImageTranspose_1x8x16(src + col * 1 + 7 * srcStride, -srcStride, dst + col * dstStride - 7, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<1>(src + col * 1 + i * srcStride, dst + col * dstStride - i * 1);
                src += 8 * srcStride;
                dst -= 8;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<1>(src + col * 1, dst + col * dstStride);
                src += srcStride;
                dst -= 1;
            }
        }

        template<> void TransformImageRotate270<2>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (height - 1) * 2;
            size_t width8 = AlignLo(width, 8);
            size_t width16 = AlignLo(width, 16);
            size_t height8 = AlignLo(height, 8);
            size_t height16 = AlignLo(height, 16);
            size_t row = 0;
            for (; row < height16; row += 16)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_2x16x8(src + col * 2 + 15 * srcStride, -srcStride, dst + col * dstStride - 30, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 16; ++i)
                        Base::CopyPixel<2>(src + col * 2 + i * srcStride, dst + col * dstStride - i * 2);
                src += 16 * srcStride;
                dst -= 32;
            }
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Avx2::TransformImageTranspose_2x8x16(src + col * 2 + 7 * srcStride, -srcStride, dst + col * dstStride - 14, dstStride);
                for (; col < width8; col += 8)
                    Sse41::TransformImageTranspose_2x8x8(src + col * 2 + 7 * srcStride, -srcStride, dst + col * dstStride - 14, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<2>(src + col * 2 + i * srcStride, dst + col * dstStride - i * 2);
                src += 8 * srcStride;
                dst -= 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<2>(src + col * 2, dst + col * dstStride);
                src += srcStride;
                dst -= 2;
            }
        }

        template<> void TransformImageRotate270<3>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            src += (height - 1) * srcStride;
            size_t width4 = AlignLo(width - 5, 4);
            size_t width8 = AlignLo(width - 9, 8);
            size_t height4 = AlignLo(height - 5, 4);
            size_t height8 = AlignLo(height - 9, 8);
            size_t row = 0;
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width4; col += 4)
                    Avx2::TransformImageTranspose_3x8x4(src + col * 3, -srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<3>(src + col * 3 - i * srcStride, dst + col * dstStride + i * 3);
                src -= 8 * srcStride;
                dst += 24;
            }
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_3x4x8(src + col * 3, -srcStride, dst + col * dstStride, dstStride);
                for (; col < width4; col += 4)
                    Sse41::TransformImageTranspose_3x4x4(src + col * 3, -srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        Base::CopyPixel<3>(src + col * 3 - i * srcStride, dst + col * dstStride + i * 3);
                src -= 4 * srcStride;
                dst += 12;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<3>(src + col * 3, dst + col * dstStride);
                src -= srcStride;
                dst += 3;
            }
       }

        template<> void TransformImageRotate270<4>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (height - 1) * 4;
            size_t width4 = AlignLo(width, 4);
            size_t width8 = AlignLo(width, 8);
            size_t height4 = AlignLo(height, 4);
            size_t height8 = AlignLo(height, 8);
            size_t row = 0;
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_4x8x8(src + col * 4 + 7 * srcStride, -srcStride, dst + col * dstStride - 28, dstStride);
                for (; col < width4; col += 4)
                    Avx2::TransformImageTranspose_4x8x4(src + col * 4 + 7 * srcStride, -srcStride, dst + col * dstStride - 28, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<4>(src + col * 4 + i * srcStride, dst + col * dstStride - i * 4);
                src += 8 * srcStride;
                dst -= 32;
            }
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_4x4x8(src + col * 4 + 3 * srcStride, -srcStride, dst + col * dstStride - 12, dstStride);
                for (; col < width4; col += 4)
                    Sse41::TransformImageTranspose_4x4x4(src + col * 4 + 3 * srcStride, -srcStride, dst + col * dstStride - 12, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        Base::CopyPixel<4>(src + col * 4 + i * srcStride, dst + col * dstStride - i * 4);
                src += 4 * srcStride;
                dst -= 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<4>(src + col * 4, dst + col * dstStride);
                src += srcStride;
                dst -= 4;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t N> void TransformImageTransposeRotate0(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride);

        template<> void TransformImageTransposeRotate0<1>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            size_t width16 = AlignLo(width, 16);
            size_t height16 = AlignLo(height, 16);
            size_t height8 = AlignLo(height, 8);
            size_t row = 0;
            for (; row < height16; row += 16)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Avx2::TransformImageTranspose_1x16x16(src + col * 1, srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 16; ++i)
                        Base::CopyPixel<1>(src + col * 1 + i * srcStride, dst + col * dstStride + i * 1);
                src += 16 * srcStride;
                dst += 16;
            }
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Sse41::TransformImageTranspose_1x8x16(src + col * 1, srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<1>(src + col * 1 + i * srcStride, dst + col * dstStride + i * 1);
                src += 8 * srcStride;
                dst += 8;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<1>(src + col * 1, dst + col * dstStride);
                src += srcStride;
                dst += 1;
            }
        }

        template<> void TransformImageTransposeRotate0<2>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            size_t width8 = AlignLo(width, 8);
            size_t width16 = AlignLo(width, 16);
            size_t height8 = AlignLo(height, 8);
            size_t height16 = AlignLo(height, 16);
            size_t row = 0;
            for (; row < height16; row += 16)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_2x16x8(src + col * 2, srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 16; ++i)
                        Base::CopyPixel<2>(src + col * 2 + i * srcStride, dst + col * dstStride + i * 2);
                src += 16 * srcStride;
                dst += 32;
            }
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Avx2::TransformImageTranspose_2x8x16(src + col * 2, srcStride, dst + col * dstStride, dstStride);
                for (; col < width8; col += 8)
                    Sse41::TransformImageTranspose_2x8x8(src + col * 2, srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<2>(src + col * 2 + i * srcStride, dst + col * dstStride + i * 2);
                src += 8 * srcStride;
                dst += 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<2>(src + col * 2, dst + col * dstStride);
                src += srcStride;
                dst += 2;
            }
        }

        template<> void TransformImageTransposeRotate0<3>(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            size_t width4 = AlignLo(width - 5, 4);
            size_t width8 = AlignLo(width - 9, 8);
            size_t height4 = AlignLo(height - 5, 4);
            size_t height8 = AlignLo(height - 9, 8);
            size_t row = 0;
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width4; col += 4)
                    Avx2::TransformImageTranspose_3x8x4(src + col * 3, srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<3>(src + col * 3 + i * srcStride, dst + col * dstStride + i * 3);
                src += 8 * srcStride;
                dst += 24;
            }
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_3x4x8(src + col * 3, srcStride, dst + col * dstStride, dstStride);
                for (; col < width4; col += 4)
                    Sse41::TransformImageTranspose_3x4x4(src + col * 3, srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        Base::CopyPixel<3>(src + col * 3 + i * srcStride, dst + col * dstStride + i * 3);
                src += 4 * srcStride;
                dst += 12;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<3>(src + col * 3, dst + col * dstStride);
                src += srcStride;
                dst += 3;
            }
        }

        template<> void TransformImageTransposeRotate0<4>(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            size_t width4 = AlignLo(width, 4);
            size_t width8 = AlignLo(width, 8);
            size_t height4 = AlignLo(height, 4);
            size_t height8 = AlignLo(height, 8);
            size_t row = 0;
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_4x8x8(src + col * 4, srcStride, dst + col * dstStride, dstStride);
                for (; col < width4; col += 4)
                    Avx2::TransformImageTranspose_4x8x4(src + col * 4, srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<4>(src + col * 4 + i * srcStride, dst + col * dstStride + i * 4);
                src += 8 * srcStride;
                dst += 32;
            }
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_4x4x8(src + col * 4, srcStride, dst + col * dstStride, dstStride);
                for (; col < width4; col += 4)
                    Sse41::TransformImageTranspose_4x4x4(src + col * 4, srcStride,  dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        Base::CopyPixel<4>(src + col * 4 + i*srcStride, dst + col * dstStride + i*4);
                src += 4*srcStride;
                dst += 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<4>(src + col * 4, dst + col * dstStride);
                src += srcStride;
                dst += 4;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t N> void TransformImageTransposeRotate90(const uint8_t * src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t * dst, ptrdiff_t dstStride)
        {
            dst += (width - A)*N;
            size_t widthA = AlignLo(width, A);
            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthDA; col += DA)
                    Avx2::TransformImageMirror64<N>(src + col * N, dst - col * N);
                for (; col < widthA; col += A)
                    Avx2::TransformImageMirror32<N>(src + col * N, dst - col * N);
                if (col < width)
                    Avx2::TransformImageMirror32<N>(src + (width - A) * N, dst - (width - A) * N);
                src += srcStride;
                dst += dstStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t N> void TransformImageTransposeRotate180(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride);

        template<> void TransformImageTransposeRotate180<1>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (width - 1) * dstStride + (height - 1) * 1;
            size_t width16 = AlignLo(width, 16);
            size_t height8 = AlignLo(height, 8);
            size_t height16 = AlignLo(height, 16);
            size_t row = 0;
            for (; row < height16; row += 16)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Avx2::TransformImageTranspose_1x16x16(src + col * 1 + 15 * srcStride, -srcStride, dst - col * dstStride - 15, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 16; ++i)
                        Base::CopyPixel<1>(src + col * 1 + i * srcStride, dst - col * dstStride - i * 1);
                src += 16 * srcStride;
                dst -= 16;
            }
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Sse41::TransformImageTranspose_1x8x16(src + col * 1 + 7 * srcStride, -srcStride, dst - col * dstStride - 7, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<1>(src + col * 1 + i * srcStride, dst - col * dstStride - i * 1);
                src += 8 * srcStride;
                dst -= 8;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<1>(src + col * 1, dst - col * dstStride);
                src += srcStride;
                dst -= 1;
            }
        }

        template<> void TransformImageTransposeRotate180<2>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (width - 1) * dstStride + (height - 1) * 2;
            size_t width8 = AlignLo(width, 8);
            size_t width16 = AlignLo(width, 16);
            size_t height8 = AlignLo(height, 8);
            size_t height16 = AlignLo(height, 16);
            size_t row = 0;
            for (; row < height16; row += 16)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_2x16x8(src + col * 2 + 15 * srcStride, -srcStride, dst - col * dstStride - 30, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 16; ++i)
                        Base::CopyPixel<2>(src + col * 2 + i * srcStride, dst - col * dstStride - i * 2);
                src += 16 * srcStride;
                dst -= 32;
            }
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width16; col += 16)
                    Avx2::TransformImageTranspose_2x8x16(src + col * 2 + 7 * srcStride, -srcStride, dst - col * dstStride - 14, -dstStride);
                for (; col < width8; col += 8)
                    Sse41::TransformImageTranspose_2x8x8(src + col * 2 + 7 * srcStride, -srcStride, dst - col * dstStride - 14, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<2>(src + col * 2 + i * srcStride, dst - col * dstStride - i * 2);
                src += 8 * srcStride;
                dst -= 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<2>(src + col * 2, dst - col * dstStride);
                src += srcStride;
                dst -= 2;
            }
        }

        template<> void TransformImageTransposeRotate180<3>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            src += (height - 1) * srcStride + (width - 1) * 3;
            size_t width4 = AlignLo(width - 5, 4);
            size_t width8 = AlignLo(width - 9, 8);
            size_t height4 = AlignLo(height - 5, 4);
            size_t height8 = AlignLo(height - 9, 8);
            size_t row = 0;
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width4; col += 4)
                    Avx2::TransformImageTranspose_3x8x4(src - col * 3 - 9, -srcStride, dst + (col + 3) * dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<3>(src - col * 3 - i * srcStride, dst + col * dstStride + i * 3);
                src -= 8 * srcStride;
                dst += 24;
            }
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_3x4x8(src - col * 3 - 21, -srcStride, dst + (col + 7) * dstStride, -dstStride);
                for (; col < width4; col += 4)
                    Sse41::TransformImageTranspose_3x4x4(src - col * 3 - 9, -srcStride, dst + (col + 3)* dstStride, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        Base::CopyPixel<3>(src - col * 3 - i * srcStride, dst + col * dstStride + i * 3);
                src -= 4 * srcStride;
                dst += 12;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<3>(src - col * 3, dst + col * dstStride);
                src -= srcStride;
                dst += 3;
            }
        }

        template<> void TransformImageTransposeRotate180<4>(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride)
        {
            dst += (width - 1) * dstStride + (height - 1) * 4;
            size_t width4 = AlignLo(width, 4);
            size_t width8 = AlignLo(width, 8);
            size_t height4 = AlignLo(height, 4);
            size_t height8 = AlignLo(height, 8);
            size_t row = 0;
            for (; row < height8; row += 8)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_4x8x8(src + col * 4 + 7 * srcStride, -srcStride, dst - col * dstStride - 28, -dstStride);
                for (; col < width4; col += 4)
                    Avx2::TransformImageTranspose_4x8x4(src + col * 4 + 7 * srcStride, -srcStride, dst - col * dstStride - 28, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 8; ++i)
                        Base::CopyPixel<4>(src + col * 4 + i * srcStride, dst - col * dstStride - i * 4);
                src += 8 * srcStride;
                dst -= 32;
            }
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width8; col += 8)
                    Avx2::TransformImageTranspose_4x4x8(src + col * 4 + 3 * srcStride, -srcStride, dst - col * dstStride - 12, -dstStride);
                for (; col < width4; col += 4)
                    Sse41::TransformImageTranspose_4x4x4(src + col * 4 + 3 * srcStride, -srcStride, dst - col * dstStride - 12, -dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        Base::CopyPixel<4>(src + col * 4 + i * srcStride, dst - col * dstStride - i * 4);
                src += 4 * srcStride;
                dst -= 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    Base::CopyPixel<4>(src + col * 4, dst - col * dstStride);
                src += srcStride;
                dst -= 4;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t N> void Init(ImageTransforms::TransformPtr transforms[8])
        {
            transforms[SimdTransformRotate90] = TransformImageRotate90<N>;
            transforms[SimdTransformRotate180] = TransformImageRotate180<N>;
            transforms[SimdTransformRotate270] = TransformImageRotate270<N>;
            transforms[SimdTransformTransposeRotate0] = TransformImageTransposeRotate0<N>;
            transforms[SimdTransformTransposeRotate90] = TransformImageTransposeRotate90<N>;
            transforms[SimdTransformTransposeRotate180] = TransformImageTransposeRotate180<N>;
        }

        ImageTransforms::ImageTransforms()
            : Sse41::ImageTransforms::ImageTransforms()
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
#endif
}

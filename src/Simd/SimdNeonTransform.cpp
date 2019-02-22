/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
        template<size_t N> SIMD_INLINE void CopyPixel(const uint8_t * src, uint8_t * dst)
        {
            for (size_t i = 0; i < N; ++i)
                dst[i] = src[i];
        }

        template<> SIMD_INLINE void CopyPixel<1>(const uint8_t * src, uint8_t * dst)
        {
            dst[0] = src[0];
        }

        template<> SIMD_INLINE void CopyPixel<2>(const uint8_t * src, uint8_t * dst)
        {
            ((uint16_t*)dst)[0] = ((uint16_t*)src)[0];
        }

        template<> SIMD_INLINE void CopyPixel<3>(const uint8_t * src, uint8_t * dst)
        {
            ((uint16_t*)dst)[0] = ((uint16_t*)src)[0];
            dst[2] = src[2];
        }

        template<> SIMD_INLINE void CopyPixel<4>(const uint8_t * src, uint8_t * dst)
        {
            ((uint32_t*)dst)[0] = ((uint32_t*)src)[0];
        }

        template<size_t N> void TransformImageRotate0(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            size_t rowSize = width * N;
            for (size_t row = 0; row < height; ++row)
            {
                memcpy(dst, src, rowSize);
                src += srcStride;
                dst += dstStride;
            }
        }

        template<size_t N> void TransformImageRotate90(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
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

        template<size_t N> SIMD_INLINE void TransformImageRotate180HA(const uint8_t * src, uint8_t * dst)
        {
            dst += (HA - 1)*N;
            for (size_t i = 0; i < HA; ++i)
                CopyPixel<N>(src + i * N, dst - i * N);
        }

        uint8x8_t K8_TURN = SIMD_VEC_SETR_PI8(7, 6, 5, 4, 3, 2, 1, 0);

        template<> SIMD_INLINE void TransformImageRotate180HA<1>(const uint8_t * src, uint8_t * dst)
        {
            uint8x8_t v = LoadHalf<false>(src);
            v = vtbl1_u8(v, K8_TURN);
            Store<false>(dst, v);
        }

        template<> SIMD_INLINE void TransformImageRotate180HA<2>(const uint8_t * src, uint8_t * dst)
        {
            uint8x8x2_t v = LoadHalf2<false>(src);
            v.val[0] = vtbl1_u8(v.val[0], K8_TURN);
            v.val[1] = vtbl1_u8(v.val[1], K8_TURN);
            Store2<false>(dst, v);
        }

        template<> SIMD_INLINE void TransformImageRotate180HA<3>(const uint8_t * src, uint8_t * dst)
        {
            uint8x8x3_t v = LoadHalf3<false>(src);
            v.val[0] = vtbl1_u8(v.val[0], K8_TURN);
            v.val[1] = vtbl1_u8(v.val[1], K8_TURN);
            v.val[2] = vtbl1_u8(v.val[2], K8_TURN);
            Store3<false>(dst, v);
        }

        template<> SIMD_INLINE void TransformImageRotate180HA<4>(const uint8_t * src, uint8_t * dst)
        {
            uint8x8x4_t v = LoadHalf4<false>(src);
            v.val[0] = vtbl1_u8(v.val[0], K8_TURN);
            v.val[1] = vtbl1_u8(v.val[1], K8_TURN);
            v.val[2] = vtbl1_u8(v.val[2], K8_TURN);
            v.val[3] = vtbl1_u8(v.val[3], K8_TURN);
            Store4<false>(dst, v);
        }

        template<size_t N> SIMD_INLINE void TransformImageRotate180DA(const uint8_t * src, uint8_t * dst)
        {
            TransformImageRotate180HA<N>(src + 0 * N * HA, dst - 0 * N * HA);
            TransformImageRotate180HA<N>(src + 1 * N * HA, dst - 1 * N * HA);
            TransformImageRotate180HA<N>(src + 2 * N * HA, dst - 2 * N * HA);
            TransformImageRotate180HA<N>(src + 3 * N * HA, dst - 3 * N * HA);
        }

        template<size_t N> void TransformImageRotate180(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            dst += (height - 1)*dstStride + (width - HA)*N;
            size_t widthHA = AlignLo(width, HA);
            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthDA; col += DA)
                    TransformImageRotate180DA<N>(src + col * N, dst - col * N);
                for (; col < widthHA; col += HA)
                    TransformImageRotate180HA<N>(src + col * N, dst - col * N);
                if(col < width)
                    TransformImageRotate180HA<N>(src + (width - HA) * N, dst - (width - HA) * N);
                src += srcStride;
                dst -= dstStride;
            }
        }

        template<size_t N> void TransformImageRotate270(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
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

        template<size_t N> void TransformImageTransposeRotate0(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst + col * dstStride);
                src += srcStride;
                dst += N;
            }
        }

        template<size_t N> void TransformImageTransposeRotate90(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            dst += (width - HA)*N;
            size_t widthHA = AlignLo(width, HA);
            size_t widthQA = AlignLo(width, QA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthQA; col += DA)
                    TransformImageRotate180DA<N>(src + col * N, dst - col * N);
                for (; col < widthHA; col += HA)
                    TransformImageRotate180HA<N>(src + col * N, dst - col * N);
                if (col < width)
                    TransformImageRotate180HA<N>(src + (width - HA) * N, dst - (width - HA) * N);
                src += srcStride;
                dst += dstStride;
            }
        }

        template<size_t N> void TransformImageTransposeRotate180(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
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

        template<size_t N> void TransformImageTransposeRotate270(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
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

        template<size_t N> void TransformImage(const uint8_t * src, size_t srcStride, size_t width, size_t height, SimdTransformType transform, uint8_t * dst, size_t dstStride)
        {
            typedef void(*TransformImagePtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);
            static const TransformImagePtr transformImage[8] = { TransformImageRotate0<N>, TransformImageRotate90<N>, TransformImageRotate180<N>, TransformImageRotate270<N>,
                TransformImageTransposeRotate0<N>, TransformImageTransposeRotate90<N>, TransformImageTransposeRotate180<N>, TransformImageTransposeRotate270<N> };
            transformImage[(int)transform](src, srcStride, width, height, dst, dstStride);
        };

        void TransformImage(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t * dst, size_t dstStride)
        {
            switch (pixelSize)
            {
            case 1: TransformImage<1>(src, srcStride, width, height, transform, dst, dstStride); break;
            case 2: TransformImage<2>(src, srcStride, width, height, transform, dst, dstStride); break;
            case 3: TransformImage<3>(src, srcStride, width, height, transform, dst, dstStride); break;
            case 4: TransformImage<4>(src, srcStride, width, height, transform, dst, dstStride); break;
            default: assert(0);
           }
        }
#endif// SIMD_NEON_ENABLE
    }
}

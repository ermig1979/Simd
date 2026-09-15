/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar,
*               2018-2018 Radchenko Andrey.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdNeural.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> SIMD_INLINE float32x4_t Pooling2x2Max2x2(const float * src, size_t stride)
        {
            float32x4_t s0 = vmaxq_f32(Load<align>(src + 0), Load<align>(src + stride + 0));
            float32x4_t s1 = vmaxq_f32(Load<align>(src + F), Load<align>(src + stride + F));
            return vcombine_f32(vpmax_f32(vget_low_f32(s0), vget_high_f32(s0)), vpmax_f32(vget_low_f32(s1), vget_high_f32(s1)));
        }

        template <bool align> SIMD_INLINE float32x4_t Pooling2x2Max2(const float * src)
        {
            float32x4_t s0 = Load<align>(src + 0);
            float32x4_t s1 = Load<align>(src + F);
            return vcombine_f32(vpmax_f32(vget_low_f32(s0), vget_high_f32(s0)), vpmax_f32(vget_low_f32(s1), vget_high_f32(s1)));
        }

        template <bool align> void NeuralPooling2x2Max2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (align)
                assert(Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F));

            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t alignedWidth = AlignLo(width, DF);
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<align>(dst + (col >> 1), Pooling2x2Max2x2<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), Pooling2x2Max2x2<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Simd::Max(src[widthEven], src[widthEven + srcStride]);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<align>(dst + (col >> 1), Pooling2x2Max2<align>(src + col));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), Pooling2x2Max2<false>(src + col));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = src[widthEven];
            }
        }

        void NeuralPooling2x2Max2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling2x2Max2x2<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling2x2Max2x2<false>(src, srcStride, width, height, dst, dstStride);
        }

        SIMD_INLINE float Max2(const float * src)
        {
            return Simd::Max(src[0], src[1]);
        }

        SIMD_INLINE float Max2x2(const float * src, size_t stride)
        {
            return Simd::Max(Max2(src), Max2(src + stride));
        }

        SIMD_INLINE float Max2x3(const float * src, size_t stride)
        {
            return Simd::Max(Max2(src), Simd::Max(Max2(src + stride), Max2(src + 2 * stride)));
        }

        template <bool align> SIMD_INLINE float32x4_t Pooling2x2Max1x3(const float * src, size_t stride)
        {
            return vmaxq_f32(vmaxq_f32(Load<align>(src), Load<align>(src + stride)), Load<align>(src + 2 * stride));
        }

        const uint8x8_t K8_TBL_BITS_LO = SIMD_VEC_SETR_PI8(0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B);
        const uint8x8_t K8_TBL_BITS_HI = SIMD_VEC_SETR_PI8(0x04, 0x05, 0x06, 0x07, 0x0C, 0x0D, 0x0E, 0x0F);

        SIMD_INLINE float32x4_t CombineFor2x2(const float32x4_t & lo, const float32x4_t & hi)
        {
            return vcombine_f32((float32x2_t)vtbl2_u8((const uint8x8x2_t &)lo, K8_TBL_BITS_LO), (float32x2_t)vtbl2_u8((const uint8x8x2_t &)hi, K8_TBL_BITS_HI));
        }

        template <bool align> SIMD_INLINE float32x4_t Pooling2x2Max3x3(const float * src, size_t stride)
        {
            float32x4_t _0123 = Pooling2x2Max1x3<align>(src, stride);
            float32x4_t _4567 = Pooling2x2Max1x3<align>(src + F, stride);
            float32x4_t _5678 = Pooling2x2Max1x3<false>(src + F + 1, stride);
            float32x4x2_t _02461357 = vuzpq_f32(_0123, _4567);
            float32x4_t _2468 = CombineFor2x2(_02461357.val[0], _5678);
            return vmaxq_f32(vmaxq_f32(_02461357.val[0], _02461357.val[1]), _2468);
        }

        template <bool align> SIMD_INLINE float32x4_t Pooling2x2Max1x2(const float * src, size_t stride)
        {
            return vmaxq_f32(Load<align>(src), Load<align>(src + stride));
        }

        template <bool align> SIMD_INLINE float32x4_t Pooling2x2Max3x2(const float * src, size_t stride)
        {
            float32x4_t _0123 = Pooling2x2Max1x2<align>(src, stride);
            float32x4_t _4567 = Pooling2x2Max1x2<align>(src + F, stride);
            float32x4_t _5678 = Pooling2x2Max1x2<false>(src + F + 1, stride);
            float32x4x2_t _02461357 = vuzpq_f32(_0123, _4567);
            float32x4_t _2468 = CombineFor2x2(_02461357.val[0], _5678);
            return vmaxq_f32(vmaxq_f32(_02461357.val[0], _02461357.val[1]), _2468);
        }

        template <bool align> void NeuralPooling2x2Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t alignedWidth = AlignLo(width, DF);
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<align>(dst + (col >> 1), Pooling2x2Max3x3<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Max2x3(src + widthEven, srcStride);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<align>(dst + (col >> 1), Pooling2x2Max3x2<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Max2x2(src + widthEven, srcStride);
            }
        }

        void NeuralPooling2x2Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling2x2Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling2x2Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}

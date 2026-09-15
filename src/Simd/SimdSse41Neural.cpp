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
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdNeural.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <bool align> SIMD_INLINE __m128 Pooling1x1Max3x1Body(const float* src)
        {
            return _mm_max_ps(_mm_max_ps(Load<false>(src - 1), Load<align>(src)), Load<false>(src + 1));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Body(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Body<align>(src - stride);
            __m128 src1 = Pooling1x1Max3x1Body<align>(src);
            __m128 src2 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, _mm_max_ps(_mm_max_ps(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Body(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Body<align>(src);
            __m128 src1 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, _mm_max_ps(src0, src1));
        }

        template <bool align> SIMD_INLINE __m128 Pooling1x1Max3x1Nose(const float* src)
        {
            __m128 src1 = Load<align>(src);
            __m128 src0 = _mm_shuffle_ps(src1, src1, 0x90);
            __m128 src2 = Load<false>(src + 1);
            return _mm_max_ps(_mm_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Nose(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Nose<align>(src - stride);
            __m128 src1 = Pooling1x1Max3x1Nose<align>(src);
            __m128 src2 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, _mm_max_ps(_mm_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Nose(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Nose<align>(src);
            __m128 src1 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, _mm_max_ps(src0, src1));
        }

        template <bool align> SIMD_INLINE __m128 Pooling1x1Max3x1Tail(const float* src)
        {
            __m128 src0 = Load<false>(src - 1);
            __m128 src1 = Load<align>(src);
            __m128 src2 = _mm_shuffle_ps(src1, src1, 0xF9);
            return _mm_max_ps(_mm_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Tail(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Tail<align>(src - stride);
            __m128 src1 = Pooling1x1Max3x1Tail<align>(src);
            __m128 src2 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, _mm_max_ps(_mm_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Tail(const float* src, size_t stride, float* dst)
        {
            __m128 src0 = Pooling1x1Max3x1Tail<align>(src);
            __m128 src1 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, _mm_max_ps(src0, src1));
        }

        template <bool align> void NeuralPooling1x1Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            assert(width > F && height > 1);

            size_t alignedWidth = AlignHi(width, F) - F;
            height -= 1;

            Pooling1x1Max3x2Nose<align>(src, srcStride, dst);
            for (size_t col = F; col < alignedWidth; col += F)
                Pooling1x1Max3x2Body<align>(src + col, srcStride, dst + col);
            Pooling1x1Max3x2Tail<false>(src + width - F, srcStride, dst + width - F);

            for (size_t row = 1; row < height; ++row)
            {
                src += srcStride;
                dst += dstStride;
                Pooling1x1Max3x3Nose<align>(src, srcStride, dst);
                for (size_t col = F; col < alignedWidth; col += F)
                    Pooling1x1Max3x3Body<align>(src + col, srcStride, dst + col);
                Pooling1x1Max3x3Tail<false>(src + width - F, srcStride, dst + width - F);
            }

            dst += dstStride;
            Pooling1x1Max3x2Nose<align>(src, srcStride, dst);
            for (size_t col = F; col < alignedWidth; col += F)
                Pooling1x1Max3x2Body<align>(src + col, srcStride, dst + col);
            Pooling1x1Max3x2Tail<false>(src + width - F, srcStride, dst + width - F);
        }

        void NeuralPooling1x1Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling1x1Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling1x1Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max2x2(const float* src, size_t stride)
        {
            __m128 _src0 = _mm_max_ps(Load<align>(src + 0), Load<align>(src + stride + 0));
            __m128 _src1 = _mm_max_ps(Load<align>(src + F), Load<align>(src + stride + F));
            return _mm_max_ps(_mm_shuffle_ps(_src0, _src1, 0x88), _mm_shuffle_ps(_src0, _src1, 0xDD));
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max2(const float* src)
        {
            __m128 _src0 = Load<align>(src + 0);
            __m128 _src1 = Load<align>(src + F);
            return _mm_max_ps(_mm_shuffle_ps(_src0, _src1, 0x88), _mm_shuffle_ps(_src0, _src1, 0xDD));
        }

        template <bool align> void NeuralPooling2x2Max2x2(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
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

        void NeuralPooling2x2Max2x2(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling2x2Max2x2<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling2x2Max2x2<false>(src, srcStride, width, height, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE float Max2(const float* src)
        {
            return Simd::Max(src[0], src[1]);
        }

        SIMD_INLINE float Max2x2(const float* src, size_t stride)
        {
            return Simd::Max(Max2(src), Max2(src + stride));
        }

        SIMD_INLINE float Max2x3(const float* src, size_t stride)
        {
            return Simd::Max(Max2(src), Simd::Max(Max2(src + stride), Max2(src + 2 * stride)));
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max1x3(const float* src, size_t stride)
        {
            return _mm_max_ps(_mm_max_ps(Load<align>(src), Load<align>(src + stride)), Load<align>(src + 2 * stride));
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max3x3(const float* src, size_t stride)
        {
            __m128 _0123 = Pooling2x2Max1x3<align>(src, stride);
            __m128 _4567 = Pooling2x2Max1x3<align>(src + F, stride);
            __m128 _5678 = Pooling2x2Max1x3<false>(src + F + 1, stride);
            __m128 _0246 = _mm_shuffle_ps(_0123, _4567, 0x88);
            __m128 _1357 = _mm_shuffle_ps(_0123, _4567, 0xDD);
            __m128 _2468 = _mm_shuffle_ps(_0246, _5678, 0xD9);
            return _mm_max_ps(_mm_max_ps(_0246, _1357), _2468);
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max1x2(const float* src, size_t stride)
        {
            return _mm_max_ps(Load<align>(src), Load<align>(src + stride));
        }

        template <bool align> SIMD_INLINE __m128 Pooling2x2Max3x2(const float* src, size_t stride)
        {
            __m128 _0123 = Pooling2x2Max1x2<align>(src, stride);
            __m128 _4567 = Pooling2x2Max1x2<align>(src + F, stride);
            __m128 _5678 = Pooling2x2Max1x2<false>(src + F + 1, stride);
            __m128 _0246 = _mm_shuffle_ps(_0123, _4567, 0x88);
            __m128 _1357 = _mm_shuffle_ps(_0123, _4567, 0xDD);
            __m128 _2468 = _mm_shuffle_ps(_0246, _5678, 0xD9);
            return _mm_max_ps(_mm_max_ps(_0246, _1357), _2468);
        }

        template <bool align> void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
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

        void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling2x2Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling2x2Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif
}

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

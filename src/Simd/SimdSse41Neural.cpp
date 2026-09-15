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
        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, const __m128& norm, const __m128& alpha, const __m128& epsilon, float* gradient, float* weight)
        {
            __m128 d = _mm_mul_ps(Load<align>(delta), norm);
            __m128 _gradient = _mm_add_ps(Load<align>(gradient), _mm_mul_ps(d, d));
            Store<align>(gradient, _gradient);
            Store<align>(weight, _mm_sub_ps(Load<align>(weight), _mm_mul_ps(_mm_mul_ps(alpha, d), _mm_rsqrt_ps(_mm_add_ps(_gradient, epsilon)))));
        }

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, size_t offset, const __m128& norm, const __m128& alpha, const __m128& epsilon, float* gradient, float* weight)
        {
            AdaptiveGradientUpdate<align>(delta + offset, norm, alpha, epsilon, gradient + offset, weight + offset);
        }

        template <bool align> void NeuralAdaptiveGradientUpdate(const float* delta, size_t size, size_t batch, const float* alpha, const float* epsilon, float* gradient, float* weight)
        {
            if (align)
                assert(Aligned(delta) && Aligned(gradient) && Aligned(weight));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            const float norm = (float)(1.0 / batch);
            __m128 _norm = _mm_set1_ps(norm);
            __m128 _alpha = _mm_set1_ps(*alpha);
            __m128 _epsilon = _mm_set1_ps(*epsilon);
            size_t i = 0;
            if (partialAlignedSize)
            {
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        AdaptiveGradientUpdate<align>(delta, i + F * 0, _norm, _alpha, _epsilon, gradient, weight);
                        AdaptiveGradientUpdate<align>(delta, i + F * 1, _norm, _alpha, _epsilon, gradient, weight);
                        AdaptiveGradientUpdate<align>(delta, i + F * 2, _norm, _alpha, _epsilon, gradient, weight);
                        AdaptiveGradientUpdate<align>(delta, i + F * 3, _norm, _alpha, _epsilon, gradient, weight);
                    }
                }
                for (; i < partialAlignedSize; i += F)
                    AdaptiveGradientUpdate<align>(delta, i, _norm, _alpha, _epsilon, gradient, weight);
            }
            for (; i < size; ++i)
                Base::AdaptiveGradientUpdate(delta, i, norm, *alpha, *epsilon, gradient, weight);
        }

        void NeuralAdaptiveGradientUpdate(const float* delta, size_t size, size_t batch, const float* alpha, const float* epsilon, float* gradient, float* weight)
        {
            if (Aligned(delta) && Aligned(gradient) && Aligned(weight))
                NeuralAdaptiveGradientUpdate<true>(delta, size, batch, alpha, epsilon, gradient, weight);
            else
                NeuralAdaptiveGradientUpdate<false>(delta, size, batch, alpha, epsilon, gradient, weight);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> void NeuralDerivativeRelu(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float s = slope[0];
            __m128 _0 = _mm_set1_ps(0.0f);
            __m128 _s = _mm_set1_ps(s);
            __m128 d = _mm_set1_ps(1.0f - s);
            size_t alignedSize = Simd::AlignLo(size, F);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 mask = _mm_cmpgt_ps(Load<align>(src + i), _0);
                __m128 _dst = Load<align>(dst + i);
                Store<align>(dst + i, _mm_mul_ps(_mm_add_ps(_s, _mm_and_ps(mask, d)), _dst));
            }
            for (; i < size; ++i)
                dst[i] *= src[i] > 0 ? 1.0f : s;
        }

        void NeuralDerivativeRelu(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeRelu<true>(src, size, slope, dst);
            else
                NeuralDerivativeRelu<false>(src, size, slope, dst);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralDerivativeTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(*slope);
            __m128 _1 = _mm_set1_ps(1.0f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 _src = Load<align>(src + i);
                __m128 _dst = Load<align>(dst + i);
                Store<align>(dst + i, _mm_mul_ps(_mm_mul_ps(_dst, _slope), _mm_sub_ps(_1, _mm_mul_ps(_src, _src))));
            }
            for (; i < size; ++i)
                dst[i] *= slope[0] * Base::DerivativeTanh(src[i]);
        }

        void NeuralDerivativeTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeTanh<true>(src, size, slope, dst);
            else
                NeuralDerivativeTanh<false>(src, size, slope, dst);
        }

        //-----------------------------------------------------------------------------------------

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

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void UpdateWeights(const float* x, const __m128& a, const __m128& b, float* d, float* w)
        {
            __m128 _d = _mm_add_ps(_mm_mul_ps(a, Load<align>(d)), _mm_mul_ps(b, Load<align>(x)));
            Store<align>(d, _d);
            Store<align>(w, _mm_add_ps(Load<align>(w), _d));
        }

        template <bool align> SIMD_INLINE void UpdateWeights(const float* x, size_t offset, const __m128& a, const __m128& b, float* d, float* w)
        {
            UpdateWeights<align>(x + offset, a, b, d + offset, w + offset);
        }

        template <bool align> void NeuralUpdateWeights(const float* x, size_t size, const float& a, const float& b, float* d, float* w)
        {
            if (align)
                assert(Aligned(x) && Aligned(d) && Aligned(w));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            __m128 _a = _mm_set1_ps(a);
            __m128 _b = _mm_set1_ps(b);
            size_t i = 0;
            if (partialAlignedSize)
            {
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        UpdateWeights<align>(x, i + F * 0, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F * 1, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F * 2, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F * 3, _a, _b, d, w);
                    }
                }
                for (; i < partialAlignedSize; i += F)
                    UpdateWeights<align>(x, i, _a, _b, d, w);
            }
            for (; i < size; ++i)
                Base::UpdateWeights(x, i, a, b, d, w);
        }

        void NeuralUpdateWeights(const float* x, size_t size, const float* a, const float* b, float* d, float* w)
        {
            if (Aligned(x) && Aligned(d) && Aligned(w))
                NeuralUpdateWeights<true>(x, size, *a, *b, d, w);
            else
                NeuralUpdateWeights<false>(x, size, *a, *b, d, w);
        }
    }
#endif
}

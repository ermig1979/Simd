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
        template <bool align> SIMD_INLINE void NeuralDerivativeTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            float32x4_t _slope = vdupq_n_f32(*slope);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                float32x4_t _src = Load<align>(src + i);
                float32x4_t _dst = Load<align>(dst + i);
                Store<align>(dst + i, vmulq_f32(vmulq_f32(_dst, _slope), vsubq_f32(_1, vmulq_f32(_src, _src))));
            }
            for (; i < size; ++i)
                dst[i] *= slope[0] * Base::DerivativeTanh(src[i]);
        }

        void NeuralDerivativeTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeTanh<true>(src, size, slope, dst);
            else
                NeuralDerivativeTanh<false>(src, size, slope, dst);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> void NeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float s = slope[0];
            float32x4_t _0 = vdupq_n_f32(0.0f);
            float32x4_t _1 = vdupq_n_f32(1.0f);
            float32x4_t _s = vdupq_n_f32(s);
            size_t alignedSize = Simd::AlignLo(size, F);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                uint32x4_t mask = vcgtq_f32(Load<align>(src + i), _0);
                float32x4_t _dst = Load<align>(dst + i);
                Store<align>(dst + i, vmulq_f32(vbslq_f32(mask, _1, _s), _dst));
            }
            for (; i < size; ++i)
                dst[i] *= src[i] > 0 ? 1.0f : s;
        }

        void NeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeRelu<true>(src, size, slope, dst);
            else
                NeuralDerivativeRelu<false>(src, size, slope, dst);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void UpdateWeights(const float * x, const float32x4_t & a, const float32x4_t & b, float * d, float * w)
        {
            float32x4_t _d = vaddq_f32(vmulq_f32(a, Load<align>(d)), vmulq_f32(b, Load<align>(x)));
            Store<align>(d, _d);
            Store<align>(w, vaddq_f32(Load<align>(w), _d));
        }

        template <bool align> SIMD_INLINE void UpdateWeights(const float * x, size_t offset, const float32x4_t & a, const float32x4_t & b, float * d, float * w)
        {
            UpdateWeights<align>(x + offset, a, b, d + offset, w + offset);
        }

        template <bool align> void NeuralUpdateWeights(const float * x, size_t size, const float & a, const float & b, float * d, float * w)
        {
            if (align)
                assert(Aligned(x) && Aligned(d) && Aligned(w));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            float32x4_t _a = vdupq_n_f32(a);
            float32x4_t _b = vdupq_n_f32(b);
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

        void NeuralUpdateWeights(const float * x, size_t size, const float * a, const float * b, float * d, float * w)
        {
            if (Aligned(x) && Aligned(d) && Aligned(w))
                NeuralUpdateWeights<true>(x, size, *a, *b, d, w);
            else
                NeuralUpdateWeights<false>(x, size, *a, *b, d, w);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float * delta, const float32x4_t & norm, const float32x4_t & alpha, const float32x4_t & epsilon, float * gradient, float * weight)
        {
            float32x4_t d = vmulq_f32(Load<align>(delta), norm);
            float32x4_t _gradient = vaddq_f32(Load<align>(gradient), vmulq_f32(d, d));
            Store<align>(gradient, _gradient);
            Store<align>(weight, vsubq_f32(Load<align>(weight), vmulq_f32(vmulq_f32(alpha, d), ReciprocalSqrt<1>(vaddq_f32(_gradient, epsilon)))));
        }

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float * delta, size_t offset, const float32x4_t & norm, const float32x4_t & alpha, const float32x4_t & epsilon, float * gradient, float * weight)
        {
            AdaptiveGradientUpdate<align>(delta + offset, norm, alpha, epsilon, gradient + offset, weight + offset);
        }

        template <bool align> void NeuralAdaptiveGradientUpdate(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight)
        {
            if (align)
                assert(Aligned(delta) && Aligned(gradient) && Aligned(weight));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            const float norm = (float)(1.0 / batch);
            float32x4_t _norm = vdupq_n_f32(norm);
            float32x4_t _alpha = vdupq_n_f32(*alpha);
            float32x4_t _epsilon = vdupq_n_f32(*epsilon);
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

        void NeuralAdaptiveGradientUpdate(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight)
        {
            if (Aligned(delta) && Aligned(gradient) && Aligned(weight))
                NeuralAdaptiveGradientUpdate<true>(delta, size, batch, alpha, epsilon, gradient, weight);
            else
                NeuralAdaptiveGradientUpdate<false>(delta, size, batch, alpha, epsilon, gradient, weight);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE float32x4_t Pooling1x1Max3x1Body(const float * src)
        {
            return vmaxq_f32(vmaxq_f32(Load<false>(src - 1), Load<align>(src)), Load<false>(src + 1));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Body(const float * src, size_t stride, float * dst)
        {
            float32x4_t src0 = Pooling1x1Max3x1Body<align>(src - stride);
            float32x4_t src1 = Pooling1x1Max3x1Body<align>(src);
            float32x4_t src2 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, vmaxq_f32(vmaxq_f32(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Body(const float * src, size_t stride, float * dst)
        {
            float32x4_t src0 = Pooling1x1Max3x1Body<align>(src);
            float32x4_t src1 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, vmaxq_f32(src0, src1));
        }

        template <bool align> SIMD_INLINE float32x4_t Pooling1x1Max3x1Nose(const float * src)
        {
            float32x4_t src1 = Load<align>(src);
            float32x4_t src0 = vextq_f32(vextq_f32(src1, src1, 1), src1, 3);
            float32x4_t src2 = Load<false>(src + 1);
            return vmaxq_f32(vmaxq_f32(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Nose(const float * src, size_t stride, float * dst)
        {
            float32x4_t src0 = Pooling1x1Max3x1Nose<align>(src - stride);
            float32x4_t src1 = Pooling1x1Max3x1Nose<align>(src);
            float32x4_t src2 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, vmaxq_f32(vmaxq_f32(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Nose(const float * src, size_t stride, float * dst)
        {
            float32x4_t src0 = Pooling1x1Max3x1Nose<align>(src);
            float32x4_t src1 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, vmaxq_f32(src0, src1));
        }

        template <bool align> SIMD_INLINE float32x4_t Pooling1x1Max3x1Tail(const float * src)
        {
            float32x4_t src0 = Load<false>(src - 1);
            float32x4_t src1 = Load<align>(src);
            float32x4_t src2 = vextq_f32(src1, vextq_f32(src1, src1, 3), 1);
            return vmaxq_f32(vmaxq_f32(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Tail(const float * src, size_t stride, float * dst)
        {
            float32x4_t src0 = Pooling1x1Max3x1Tail<align>(src - stride);
            float32x4_t src1 = Pooling1x1Max3x1Tail<align>(src);
            float32x4_t src2 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, vmaxq_f32(vmaxq_f32(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Tail(const float * src, size_t stride, float * dst)
        {
            float32x4_t src0 = Pooling1x1Max3x1Tail<align>(src);
            float32x4_t src1 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, vmaxq_f32(src0, src1));
        }

        template <bool align> void NeuralPooling1x1Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
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

        void NeuralPooling1x1Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralPooling1x1Max3x3<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralPooling1x1Max3x3<false>(src, srcStride, width, height, dst, dstStride);
        }

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

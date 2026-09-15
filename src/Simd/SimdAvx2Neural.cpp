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
#include "Simd/SimdBase.h"
#include "Simd/SimdNeural.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, const __m256& norm, const __m256& alpha, const __m256& epsilon, float* gradient, float* weight)
        {
            __m256 d = _mm256_mul_ps(Load<align>(delta), norm);
            __m256 _gradient = _mm256_add_ps(Load<align>(gradient), _mm256_mul_ps(d, d));
            Store<align>(gradient, _gradient);
            Store<align>(weight, _mm256_sub_ps(Load<align>(weight), _mm256_mul_ps(_mm256_mul_ps(alpha, d), _mm256_rsqrt_ps(_mm256_add_ps(_gradient, epsilon)))));
        }

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, size_t offset, const __m256& norm, const __m256& alpha, const __m256& epsilon, float* gradient, float* weight)
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
            __m256 _norm = _mm256_set1_ps(norm);
            __m256 _alpha = _mm256_set1_ps(*alpha);
            __m256 _epsilon = _mm256_set1_ps(*epsilon);
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

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE __m256 Pooling1x1Max3x1Body(const float * src)
        {
            return _mm256_max_ps(_mm256_max_ps(Load<false>(src - 1), Load<align>(src)), Load<false>(src + 1));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Body(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Body<align>(src - stride);
            __m256 src1 = Pooling1x1Max3x1Body<align>(src);
            __m256 src2 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, _mm256_max_ps(_mm256_max_ps(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Body(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Body<align>(src);
            __m256 src1 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, _mm256_max_ps(src0, src1));
        }

        __m256i K32_PERMUTE_NOSE = SIMD_MM256_SETR_EPI32(0, 0, 1, 2, 3, 4, 5, 6);

        template <bool align> SIMD_INLINE __m256 Pooling1x1Max3x1Nose(const float * src)
        {
            __m256 src1 = Load<align>(src);
            __m256 src0 = _mm256_permutevar8x32_ps(src1, K32_PERMUTE_NOSE);
            __m256 src2 = Load<false>(src + 1);
            return _mm256_max_ps(_mm256_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Nose(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Nose<align>(src - stride);
            __m256 src1 = Pooling1x1Max3x1Nose<align>(src);
            __m256 src2 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, _mm256_max_ps(_mm256_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Nose(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Nose<align>(src);
            __m256 src1 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, _mm256_max_ps(src0, src1));
        }

        __m256i K32_PERMUTE_TAIL = SIMD_MM256_SETR_EPI32(1, 2, 3, 4, 5, 6, 7, 7);

        template <bool align> SIMD_INLINE __m256 Pooling1x1Max3x1Tail(const float * src)
        {
            __m256 src0 = Load<false>(src - 1);
            __m256 src1 = Load<align>(src);
            __m256 src2 = _mm256_permutevar8x32_ps(src1, K32_PERMUTE_TAIL);
            return _mm256_max_ps(_mm256_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Tail(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Tail<align>(src - stride);
            __m256 src1 = Pooling1x1Max3x1Tail<align>(src);
            __m256 src2 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, _mm256_max_ps(_mm256_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Tail(const float * src, size_t stride, float * dst)
        {
            __m256 src0 = Pooling1x1Max3x1Tail<align>(src);
            __m256 src1 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, _mm256_max_ps(src0, src1));
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

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max2x2(const float* src, size_t stride)
        {
            __m256 lo = _mm256_max_ps(Load<align>(src + 0), Load<align>(src + stride + 0));
            __m256 hi = _mm256_max_ps(Load<align>(src + F), Load<align>(src + stride + F));
            __m256 _lo = _mm256_permute2f128_ps(lo, hi, 0x20);
            __m256 _hi = _mm256_permute2f128_ps(lo, hi, 0x31);
            return _mm256_max_ps(_mm256_shuffle_ps(_lo, _hi, 0x88), _mm256_shuffle_ps(_lo, _hi, 0xDD));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max2(const float* src)
        {
            __m256 lo = Load<align>(src + 0);
            __m256 hi = Load<align>(src + F);
            __m256 _lo = _mm256_permute2f128_ps(lo, hi, 0x20);
            __m256 _hi = _mm256_permute2f128_ps(lo, hi, 0x31);
            return _mm256_max_ps(_mm256_shuffle_ps(_lo, _hi, 0x88), _mm256_shuffle_ps(_lo, _hi, 0xDD));
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

        //-------------------------------------------------------------------------------------------------

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

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max1x3(const float * src, size_t stride)
        {
            return _mm256_max_ps(_mm256_max_ps(Load<align>(src), Load<align>(src + stride)), Load<align>(src + 2 * stride));
        }

        SIMD_INLINE __m256 PermuteFor2x2(__m256 a)
        {
            return _mm256_castsi256_ps(_mm256_permute4x64_epi64(_mm256_castps_si256(a), 0xD8));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max3x3(const float * src, size_t stride)
        {
            __m256 _01234567 = Pooling2x2Max1x3<align>(src, stride);
            __m256 _89abcdef = Pooling2x2Max1x3<align>(src + F, stride);
            __m256 _456789ab = _mm256_permute2f128_ps(_01234567, _89abcdef, 0x21);
            __m256 _12345678 = Alignr<1>(_01234567, _456789ab);
            __m256 _9abcdefg = Pooling2x2Max1x3<false>(src + F + 1, stride);
            __m256 _028a46ce = _mm256_shuffle_ps(_01234567, _89abcdef, 0x88);
            __m256 _139b57df = _mm256_shuffle_ps(_01234567, _89abcdef, 0xDD);
            __m256 _24ac68eg = _mm256_shuffle_ps(_12345678, _9abcdefg, 0xDD);
            return PermuteFor2x2(_mm256_max_ps(_mm256_max_ps(_028a46ce, _139b57df), _24ac68eg));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max1x2(const float * src, size_t stride)
        {
            return _mm256_max_ps(Load<align>(src), Load<align>(src + stride));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max3x2(const float * src, size_t stride)
        {
            __m256 _01234567 = Pooling2x2Max1x2<align>(src, stride);
            __m256 _89abcdef = Pooling2x2Max1x2<align>(src + F, stride);
            __m256 _456789ab = _mm256_permute2f128_ps(_01234567, _89abcdef, 0x21);
            __m256 _12345678 = Alignr<1>(_01234567, _456789ab);
            __m256 _9abcdefg = Pooling2x2Max1x2<false>(src + F + 1, stride);
            __m256 _028a46ce = _mm256_shuffle_ps(_01234567, _89abcdef, 0x88);
            __m256 _139b57df = _mm256_shuffle_ps(_01234567, _89abcdef, 0xDD);
            __m256 _24ac68eg = _mm256_shuffle_ps(_12345678, _9abcdefg, 0xDD);
            return PermuteFor2x2(_mm256_max_ps(_mm256_max_ps(_028a46ce, _139b57df), _24ac68eg));
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
#endif// SIMD_AVX2_ENABLE
}

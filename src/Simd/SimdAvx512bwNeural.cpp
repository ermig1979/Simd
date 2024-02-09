/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdStream.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdNeural.h"
#include "Simd/SimdPow.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void AddValue(const __m512& value, float* dst, __mmask16 m = -1)
        {
            __m512 _dst = Load<align, mask>(dst, m);
            Store<align, mask>(dst, _mm512_add_ps(_dst, value), m);
        }

        template <bool align> SIMD_INLINE void AddValue(const float* value, float* dst, size_t aligned, size_t partial, size_t full)
        {
            size_t i = 0;
            __m512 _value = _mm512_set1_ps(value[0]);
            for (; i < aligned; i += QF)
            {
                AddValue<align, false>(_value, dst + i + F * 0);
                AddValue<align, false>(_value, dst + i + F * 1);
                AddValue<align, false>(_value, dst + i + F * 2);
                AddValue<align, false>(_value, dst + i + F * 3);
            }
            for (; i < partial; i += F)
                AddValue<align, false>(_value, dst + i);
            if (i < full)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - full);
                AddValue<align, true>(_value, dst + i, tailMask);
            }
        }

        void NeuralAddValue(const float* value, float* dst, size_t size)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(dst))
                AddValue<true>(value, dst, aligned, partial, size);
            else
                AddValue<false>(value, dst, aligned, partial, size);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void AddVector(const float* src, float* dst, __mmask16 m = -1)
        {
            __m512 _src = Load<align, mask>(src, m);
            __m512 _dst = Load<align, mask>(dst, m);
            Store<align, mask>(dst, _mm512_add_ps(_src, _dst), m);
        }

        template <bool align> SIMD_INLINE void AddVector(const float* src, size_t aligned, size_t partial, size_t full, float* dst)
        {
            size_t i = 0;
            for (; i < aligned; i += QF)
            {
                AddVector<align, false>(src + i + F * 0, dst + i + F * 0);
                AddVector<align, false>(src + i + F * 1, dst + i + F * 1);
                AddVector<align, false>(src + i + F * 2, dst + i + F * 2);
                AddVector<align, false>(src + i + F * 3, dst + i + F * 3);
            }
            for (; i < partial; i += F)
                AddVector<align, false>(src + i, dst + i);
            if (i < full)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - full);
                AddVector<align, true>(src + i, dst + i, tailMask);
            }
        }

        void NeuralAddVector(const float* src, size_t size, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddVector<true>(src, aligned, partial, size, dst);
            else
                AddVector<false>(src, aligned, partial, size, dst);
        }

        //-----------------------------------------------------------------------------------------

        void NeuralAddVectorMultipliedByValue(const float* src, size_t size, const float* value, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddMultiplied<true>(src, aligned, partial, size, *value, dst);
            else
                AddMultiplied<false>(src, aligned, partial, size, *value, dst);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, const __m512& norm, const __m512& alpha, const __m512& epsilon, float* gradient, float* weight, __mmask16 m)
        {
            __m512 _delta = Load<align, mask>(delta, m);
            __m512 d = _mm512_mul_ps(_delta, norm);
            __m512 _gradient = Load<align, mask>(gradient, m);
            _gradient = _mm512_fmadd_ps(d, d, _gradient);
            Store<align, mask>(gradient, _gradient, m);
            __m512 _weight = Load<align, mask>(weight, m);
            Store<align, mask>(weight, _mm512_sub_ps(_weight, _mm512_mul_ps(_mm512_mul_ps(alpha, d), Rsqrt14(_mm512_add_ps(_gradient, epsilon)))), m);
        }

        template <bool align, bool mask> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, size_t offset, const __m512& norm, const __m512& alpha, const __m512& epsilon, float* gradient, float* weight, __mmask16 m = -1)
        {
            AdaptiveGradientUpdate<align, mask>(delta + offset, norm, alpha, epsilon, gradient + offset, weight + offset, m);
        }

        template <bool align> void NeuralAdaptiveGradientUpdate(const float* delta, size_t size, size_t batch, const float* alpha, const float* epsilon, float* gradient, float* weight)
        {
            if (align)
                assert(Aligned(delta) && Aligned(gradient) && Aligned(weight));

            const float norm = (float)(1.0 / batch);
            __m512 _norm = _mm512_set1_ps(norm);
            __m512 _alpha = _mm512_set1_ps(*alpha);
            __m512 _epsilon = _mm512_set1_ps(*epsilon);
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                AdaptiveGradientUpdate<align, false>(delta, i + F * 0, _norm, _alpha, _epsilon, gradient, weight);
                AdaptiveGradientUpdate<align, false>(delta, i + F * 1, _norm, _alpha, _epsilon, gradient, weight);
                AdaptiveGradientUpdate<align, false>(delta, i + F * 2, _norm, _alpha, _epsilon, gradient, weight);
                AdaptiveGradientUpdate<align, false>(delta, i + F * 3, _norm, _alpha, _epsilon, gradient, weight);
            }
            for (; i < partialAlignedSize; i += F)
                AdaptiveGradientUpdate<align, false>(delta, i, _norm, _alpha, _epsilon, gradient, weight);
            if (i < size)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - size);
                AdaptiveGradientUpdate<align, true>(delta, i, _norm, _alpha, _epsilon, gradient, weight, tailMask);
            }
        }

        void NeuralAdaptiveGradientUpdate(const float* delta, size_t size, size_t batch, const float* alpha, const float* epsilon, float* gradient, float* weight)
        {
            if (Aligned(delta) && Aligned(gradient) && Aligned(weight))
                NeuralAdaptiveGradientUpdate<true>(delta, size, batch, alpha, epsilon, gradient, weight);
            else
                NeuralAdaptiveGradientUpdate<false>(delta, size, batch, alpha, epsilon, gradient, weight);
        }

        //-----------------------------------------------------------------------------------------

        template <bool inversion> __m128i Invert(__m128i value);

        template <> __m128i Invert<true>(__m128i value)
        {
            return _mm_sub_epi8(Sse41::K_INV_ZERO, value);
        }

        template <> __m128i Invert<false>(__m128i value)
        {
            return value;
        }

        template <bool inversion, bool align, bool stream> void Convert(const uint8_t* src, const __m512& _1_255, float* dst)
        {
            __m128i _src = Invert<inversion>(Sse41::Load<align>((__m128i*)src));
            Stream<align, stream>(dst, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_src)), _1_255));
        }

        template <bool inversion, bool align, bool stream> void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            assert(width >= F);
            if (align)
                assert(Aligned(src, Sse41::A) && Aligned(srcStride, Sse41::A) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, F);
            __m512 _1_255 = _mm512_set1_ps(1.0f / 255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                    Convert<inversion, align, stream>(src + col, _1_255, dst + col);
                if (width != alignedWidth)
                    Convert<inversion, false, stream>(src + width - F, _1_255, dst + width - F);
                src += srcStride;
                dst += dstStride;
            }
            if (stream)
                _mm_mfence();
        }

        template <bool inversion> void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            if (Aligned(src, Sse41::A) && Aligned(srcStride, Sse41::A) && Aligned(dst) && Aligned(dstStride))
            {
                if (width * height * sizeof(float) >= STREAM_SIZE_MIN)
                    NeuralConvert<inversion, true, true>(src, srcStride, width, height, dst, dstStride);
                else
                    NeuralConvert<inversion, true, false>(src, srcStride, width, height, dst, dstStride);
            }
            else
                NeuralConvert<inversion, false, false>(src, srcStride, width, height, dst, dstStride);
        }

        void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride, int inversion)
        {
            if (inversion)
                NeuralConvert<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralConvert<false>(src, srcStride, width, height, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void NeuralDerivativeRelu(const float* src, const __m512& _0, const __m512& _1, const __m512& slope, float* dst, __mmask16 m = -1)
        {
            __m512 _src = Load<align, mask>(src, m);
            __mmask16 positive = _mm512_cmp_ps_mask(_src, _0, _CMP_GT_OS);
            __m512 _dst = Load<align, mask>(dst, m);
            Store<align, mask>(dst, _mm512_mul_ps(_mm512_mask_blend_ps(positive, slope, _1), _dst), m);
        }

        template <bool align> SIMD_INLINE void NeuralDerivativeRelu(const float* src, size_t size, const float* slope, float* dst)
        {
            __m512 _0 = _mm512_set1_ps(0.0f);
            __m512 _1 = _mm512_set1_ps(1.0f);
            __m512 _slope = _mm512_set1_ps(slope[0]);
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                NeuralDerivativeRelu<align, true>(src + i + 0 * F, _0, _1, _slope, dst + i + 0 * F);
                NeuralDerivativeRelu<align, true>(src + i + 1 * F, _0, _1, _slope, dst + i + 1 * F);
                NeuralDerivativeRelu<align, true>(src + i + 2 * F, _0, _1, _slope, dst + i + 2 * F);
                NeuralDerivativeRelu<align, true>(src + i + 3 * F, _0, _1, _slope, dst + i + 3 * F);
            }
            for (; i < partialAlignedSize; i += F)
                NeuralDerivativeRelu<align, true>(src + i, _0, _1, _slope, dst + i);
            if (i < size)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - size);
                NeuralDerivativeRelu<align, true>(src + i, _0, _1, _slope, dst + i, tailMask);
            }
        }

        void NeuralDerivativeRelu(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeRelu<true>(src, size, slope, dst);
            else
                NeuralDerivativeRelu<false>(src, size, slope, dst);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void NeuralDerivativeSigmoid(const float* src, const __m512& _1, const __m512& slope, float* dst, __mmask16 m = -1)
        {
            __m512 _src = Load<align, mask>(src, m);
            __m512 _dst = Load<align, mask>(dst, m);
            Store<align, mask>(dst, _mm512_mul_ps(_mm512_mul_ps(_dst, slope), _mm512_mul_ps(_mm512_sub_ps(_1, _src), _src)), m);
        }

        template <bool align> SIMD_INLINE void NeuralDerivativeSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            __m512 _1 = _mm512_set1_ps(1.0f);
            __m512 _slope = _mm512_set1_ps(*slope);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                NeuralDerivativeSigmoid<align, true>(src + i + 0 * F, _1, _slope, dst + i + 0 * F);
                NeuralDerivativeSigmoid<align, true>(src + i + 1 * F, _1, _slope, dst + i + 1 * F);
                NeuralDerivativeSigmoid<align, true>(src + i + 2 * F, _1, _slope, dst + i + 2 * F);
                NeuralDerivativeSigmoid<align, true>(src + i + 3 * F, _1, _slope, dst + i + 3 * F);
            }
            for (; i < partialAlignedSize; i += F)
                NeuralDerivativeSigmoid<align, true>(src + i, _1, _slope, dst + i);
            if (i < size)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - size);
                NeuralDerivativeSigmoid<align, true>(src + i, _1, _slope, dst + i, tailMask);
            }
        }

        void NeuralDerivativeSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeSigmoid<true>(src, size, slope, dst);
            else
                NeuralDerivativeSigmoid<false>(src, size, slope, dst);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void NeuralDerivativeTanh(const float* src, const __m512& _1, const __m512& slope, float* dst, __mmask16 m = -1)
        {
            __m512 _src = Load<align, mask>(src, m);
            __m512 _dst = Load<align, mask>(dst, m);
            Store<align, mask>(dst, _mm512_mul_ps(_mm512_mul_ps(_dst, slope), _mm512_sub_ps(_1, _mm512_mul_ps(_src, _src))), m);
        }

        template <bool align> SIMD_INLINE void NeuralDerivativeTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            __m512 _1 = _mm512_set1_ps(1.0f);
            __m512 _slope = _mm512_set1_ps(*slope);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                NeuralDerivativeTanh<align, true>(src + i + 0 * F, _1, _slope, dst + i + 0 * F);
                NeuralDerivativeTanh<align, true>(src + i + 1 * F, _1, _slope, dst + i + 1 * F);
                NeuralDerivativeTanh<align, true>(src + i + 2 * F, _1, _slope, dst + i + 2 * F);
                NeuralDerivativeTanh<align, true>(src + i + 3 * F, _1, _slope, dst + i + 3 * F);
            }
            for (; i < partialAlignedSize; i += F)
                NeuralDerivativeTanh<align, true>(src + i, _1, _slope, dst + i);
            if (i < size)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - size);
                NeuralDerivativeTanh<align, true>(src + i, _1, _slope, dst + i, tailMask);
            }
        }

        void NeuralDerivativeTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeTanh<true>(src, size, slope, dst);
            else
                NeuralDerivativeTanh<false>(src, size, slope, dst);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE __m512 Pooling1x1Max3x1Body(const float* src)
        {
            return _mm512_max_ps(_mm512_max_ps(Load<false>(src - 1), Load<align>(src)), Load<false>(src + 1));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Body(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Body<align>(src - stride);
            __m512 src1 = Pooling1x1Max3x1Body<align>(src);
            __m512 src2 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Body(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Body<align>(src);
            __m512 src1 = Pooling1x1Max3x1Body<align>(src + stride);
            Store<align>(dst, _mm512_max_ps(src0, src1));
        }

        __m512i K32_PERMUTE_NOSE = SIMD_MM512_SETR_EPI32(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

        template <bool align> SIMD_INLINE __m512 Pooling1x1Max3x1Nose(const float* src)
        {
            __m512 src1 = Load<align>(src);
            __m512 src0 = _mm512_permutexvar_ps(K32_PERMUTE_NOSE, src1);
            __m512 src2 = Load<false>(src + 1);
            return _mm512_max_ps(_mm512_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Nose(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Nose<align>(src - stride);
            __m512 src1 = Pooling1x1Max3x1Nose<align>(src);
            __m512 src2 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Nose(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Nose<align>(src);
            __m512 src1 = Pooling1x1Max3x1Nose<align>(src + stride);
            Store<align>(dst, _mm512_max_ps(src0, src1));
        }

        __m512i K32_PERMUTE_TAIL = SIMD_MM512_SETR_EPI32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15);

        template <bool align> SIMD_INLINE __m512 Pooling1x1Max3x1Tail(const float* src)
        {
            __m512 src0 = Load<false>(src - 1);
            __m512 src1 = Load<align>(src);
            __m512 src2 = _mm512_permutexvar_ps(K32_PERMUTE_TAIL, src1);
            return _mm512_max_ps(_mm512_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Tail(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Tail<align>(src - stride);
            __m512 src1 = Pooling1x1Max3x1Tail<align>(src);
            __m512 src2 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Tail(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Tail<align>(src);
            __m512 src1 = Pooling1x1Max3x1Tail<align>(src + stride);
            Store<align>(dst, _mm512_max_ps(src0, src1));
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

        __m512i K32_PERMUTE_2_0 = SIMD_MM512_SETR_EPI32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        __m512i K32_PERMUTE_2_1 = SIMD_MM512_SETR_EPI32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        __m512i K32_PERMUTE_2_2 = SIMD_MM512_SETR_EPI32(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0);

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max2x2(const float* src, size_t stride)
        {
            __m512 lo = _mm512_max_ps(Load<align>(src + 0), Load<align>(src + stride + 0));
            __m512 hi = _mm512_max_ps(Load<align>(src + F), Load<align>(src + stride + F));
            __m512 _lo = _mm512_shuffle_f32x4(lo, hi, 0x88);
            __m512 _hi = _mm512_shuffle_f32x4(lo, hi, 0xDD);
            return _mm512_max_ps(_mm512_shuffle_ps(_lo, _hi, 0x88), _mm512_shuffle_ps(_lo, _hi, 0xDD));
        }

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max2(const float* src)
        {
            __m512 lo = Load<align>(src + 0);
            __m512 hi = Load<align>(src + F);
            __m512 _lo = _mm512_shuffle_f32x4(lo, hi, 0x88);
            __m512 _hi = _mm512_shuffle_f32x4(lo, hi, 0xDD);
            return _mm512_max_ps(_mm512_shuffle_ps(_lo, _hi, 0x88), _mm512_shuffle_ps(_lo, _hi, 0xDD));
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

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max1x3(const float* src, size_t stride)
        {
            return _mm512_max_ps(_mm512_max_ps(Load<align>(src), Load<align>(src + stride)), Load<align>(src + 2 * stride));
        }

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max3x3(const float* src, size_t stride)
        {
            __m512 s0 = Pooling2x2Max1x3<align>(src + 0, stride);
            __m512 sf = Pooling2x2Max1x3<align>(src + F, stride);
            __m512 p0 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_0, sf);
            __m512 p1 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_1, sf);
            __m512 p2 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_2, sf);
            return _mm512_max_ps(_mm512_max_ps(p0, p1), p2);
        }

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max1x2(const float* src, size_t stride)
        {
            return _mm512_max_ps(Load<align>(src), Load<align>(src + stride));
        }

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max3x2(const float* src, size_t stride)
        {
            __m512 s0 = Pooling2x2Max1x2<align>(src + 0, stride);
            __m512 sf = Pooling2x2Max1x2<align>(src + F, stride);
            __m512 p0 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_0, sf);
            __m512 p1 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_1, sf);
            __m512 p2 = _mm512_permutex2var_ps(s0, K32_PERMUTE_2_2, sf);
            return _mm512_max_ps(_mm512_max_ps(p0, p1), p2);
        }

        template <bool align> void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t step = DF - 2;
            size_t alignedWidth = width / step * step;
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += step)
                    Store<false, true>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride), __mmask16(0x7FFF));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - step;
                    Store<false, true>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride), __mmask16(0x7FFF));
                }
                if (width - widthEven)
                    Sse41::Max2x3s(src + widthEven, srcStride, dst + (widthEven >> 1));
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += step)
                    Store<false, true>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride), __mmask16(0x7FFF));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - step;
                    Store<false, true>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride), __mmask16(0x7FFF));
                }
                if (width - widthEven)
                    Sse41::Max2x2s(src + widthEven, srcStride, dst + (widthEven >> 1));
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

        template<bool align> void NeuralPow(const float* src, size_t size, const float* exponent, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            float e = exponent[0];
            size_t aligned = AlignLo(size, F);
            __m512 _e = _mm512_set1_ps(e);
            Pow pow;
            size_t i = 0;
            for (; i < aligned; i += F)
                Store<align>(dst + i, pow(Load<align>(src + i), _e));
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                Store<align, true>(dst + i, pow(Load<align, true>(src + i, tail), _e), tail);
            }
        }

        void NeuralPow(const float* src, size_t size, const float* exponent, float* dst)
        {
#if defined(_MSC_VER) && _MSC_VER <= 1912
            Avx2::NeuralPow(src, size, exponent, dst);
#else            
            if (Aligned(src) && Aligned(dst))
                NeuralPow<true>(src, size, exponent, dst);
            else
                NeuralPow<false>(src, size, exponent, dst);
#endif        
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void NeuralProductSum(const float* a, const float* b, size_t offset, __m512& sum, __mmask16 m = -1)
        {
            __m512 _a = Load<align, mask>(a + offset, m);
            __m512 _b = Load<align, mask>(b + offset, m);
            sum = _mm512_fmadd_ps(_a, _b, sum);
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float* a, const float* b, size_t size, float* sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            __m512 sum0 = _mm512_setzero_ps();
            if (fullAlignedSize)
            {
                __m512 sum1 = _mm512_setzero_ps();
                __m512 sum2 = _mm512_setzero_ps();
                __m512 sum3 = _mm512_setzero_ps();
                for (; i < fullAlignedSize; i += QF)
                {
                    NeuralProductSum<align, false>(a, b, i + F * 0, sum0);
                    NeuralProductSum<align, false>(a, b, i + F * 1, sum1);
                    NeuralProductSum<align, false>(a, b, i + F * 2, sum2);
                    NeuralProductSum<align, false>(a, b, i + F * 3, sum3);
                }
                sum0 = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
            }
            for (; i < partialAlignedSize; i += F)
                NeuralProductSum<align, false>(a, b, i, sum0);
            if (i < size)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - size);
                NeuralProductSum<align, true>(a, b, i, sum0, tailMask);
            }
            *sum = ExtractSum(sum0);
        }

        void NeuralProductSum(const float* a, const float* b, size_t size, float* sum)
        {
            if (Aligned(a) && Aligned(b))
                NeuralProductSum<true>(a, b, size, sum);
            else
                NeuralProductSum<false>(a, b, size, sum);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void NeuralUpdateWeights(const float* x, const __m512& a, const __m512& b, float* d, float* w, __mmask16 m)
        {
            __m512 _x = Load<align, mask>(x, m);
            __m512 _d = Load<align, mask>(d, m);
            _d = _mm512_fmadd_ps(a, _d, _mm512_mul_ps(b, _x));
            Store<align, mask>(d, _d, m);
            __m512 _w = Load<align, mask>(w, m);
            Store<align, mask>(w, _mm512_add_ps(_w, _d), m);
        }

        template <bool align, bool mask> SIMD_INLINE void NeuralUpdateWeights(const float* x, size_t offset, const __m512& a, const __m512& b, float* d, float* w, __mmask16 m = -1)
        {
            NeuralUpdateWeights<align, mask>(x + offset, a, b, d + offset, w + offset, m);
        }

        template <bool align> SIMD_INLINE void NeuralUpdateWeights(const float* x, size_t size, const float& a, const float& b, float* d, float* w)
        {
            if (align)
                assert(Aligned(x) && Aligned(d) && Aligned(w));

            __m512 _a = _mm512_set1_ps(a);
            __m512 _b = _mm512_set1_ps(b);
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                NeuralUpdateWeights<align, false>(x, i + F * 0, _a, _b, d, w);
                NeuralUpdateWeights<align, false>(x, i + F * 1, _a, _b, d, w);
                NeuralUpdateWeights<align, false>(x, i + F * 2, _a, _b, d, w);
                NeuralUpdateWeights<align, false>(x, i + F * 3, _a, _b, d, w);
            }
            for (; i < partialAlignedSize; i += F)
                NeuralUpdateWeights<align, false>(x, i, _a, _b, d, w);
            if (i < size)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - size);
                NeuralUpdateWeights<align, true>(x, i, _a, _b, d, w, tailMask);
            }
        }

        void NeuralUpdateWeights(const float* x, size_t size, const float* a, const float* b, float* d, float* w)
        {
            if (Aligned(x) && Aligned(d) && Aligned(w))
                NeuralUpdateWeights<true>(x, size, *a, *b, d, w);
            else
                NeuralUpdateWeights<false>(x, size, *a, *b, d, w);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

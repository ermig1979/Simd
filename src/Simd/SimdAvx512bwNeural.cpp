/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, const __m512& norm, const __m512& alpha, const __m512& epsilon, float* gradient, float* weight, __mmask16 m)
        {
            __m512 _delta = Avx512f::Load<align, mask>(delta, m);
            __m512 d = _mm512_mul_ps(_delta, norm);
            __m512 _gradient = Avx512f::Load<align, mask>(gradient, m);
            _gradient = _mm512_fmadd_ps(d, d, _gradient);
            Avx512f::Store<align, mask>(gradient, _gradient, m);
            __m512 _weight = Avx512f::Load<align, mask>(weight, m);
            Avx512f::Store<align, mask>(weight, _mm512_sub_ps(_weight, _mm512_mul_ps(_mm512_mul_ps(alpha, d), Avx512f::Rsqrt14(_mm512_add_ps(_gradient, epsilon)))), m);
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
            return _mm_sub_epi8(Sse2::K_INV_ZERO, value);
        }

        template <> __m128i Invert<false>(__m128i value)
        {
            return value;
        }

        template <bool inversion, bool align, bool stream> void Convert(const uint8_t* src, const __m512& _1_255, float* dst)
        {
            __m128i _src = Invert<inversion>(Sse2::Load<align>((__m128i*)src));
            Avx512f::Stream<align, stream>(dst, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_src)), _1_255));
        }

        template <bool inversion, bool align, bool stream> void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            assert(width >= F);
            if (align)
                assert(Aligned(src, Sse2::A) && Aligned(srcStride, Sse2::A) && Aligned(dst) && Aligned(dstStride));

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
            if (Aligned(src, Sse2::A) && Aligned(srcStride, Sse2::A) && Aligned(dst) && Aligned(dstStride))
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

        template <bool align> SIMD_INLINE __m512 Pooling1x1Max3x1Body(const float* src)
        {
            return _mm512_max_ps(_mm512_max_ps(Avx512f::Load<false>(src - 1), Avx512f::Load<align>(src)), Avx512f::Load<false>(src + 1));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Body(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Body<align>(src - stride);
            __m512 src1 = Pooling1x1Max3x1Body<align>(src);
            __m512 src2 = Pooling1x1Max3x1Body<align>(src + stride);
            Avx512f::Store<align>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Body(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Body<align>(src);
            __m512 src1 = Pooling1x1Max3x1Body<align>(src + stride);
            Avx512f::Store<align>(dst, _mm512_max_ps(src0, src1));
        }

        __m512i K32_PERMUTE_NOSE = SIMD_MM512_SETR_EPI32(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

        template <bool align> SIMD_INLINE __m512 Pooling1x1Max3x1Nose(const float* src)
        {
            __m512 src1 = Avx512f::Load<align>(src);
            __m512 src0 = _mm512_permutexvar_ps(K32_PERMUTE_NOSE, src1);
            __m512 src2 = Avx512f::Load<false>(src + 1);
            return _mm512_max_ps(_mm512_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Nose(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Nose<align>(src - stride);
            __m512 src1 = Pooling1x1Max3x1Nose<align>(src);
            __m512 src2 = Pooling1x1Max3x1Nose<align>(src + stride);
            Avx512f::Store<align>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }
        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Nose(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Nose<align>(src);
            __m512 src1 = Pooling1x1Max3x1Nose<align>(src + stride);
            Avx512f::Store<align>(dst, _mm512_max_ps(src0, src1));
        }

        __m512i K32_PERMUTE_TAIL = SIMD_MM512_SETR_EPI32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15);

        template <bool align> SIMD_INLINE __m512 Pooling1x1Max3x1Tail(const float* src)
        {
            __m512 src0 = Avx512f::Load<false>(src - 1);
            __m512 src1 = Avx512f::Load<align>(src);
            __m512 src2 = _mm512_permutexvar_ps(K32_PERMUTE_TAIL, src1);
            return _mm512_max_ps(_mm512_max_ps(src0, src1), src2);
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x3Tail(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Tail<align>(src - stride);
            __m512 src1 = Pooling1x1Max3x1Tail<align>(src);
            __m512 src2 = Pooling1x1Max3x1Tail<align>(src + stride);
            Avx512f::Store<align>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }

        template <bool align> SIMD_INLINE void Pooling1x1Max3x2Tail(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = Pooling1x1Max3x1Tail<align>(src);
            __m512 src1 = Pooling1x1Max3x1Tail<align>(src + stride);
            Avx512f::Store<align>(dst, _mm512_max_ps(src0, src1));
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
            __m512 lo = _mm512_max_ps(Avx512f::Load<align>(src + 0), Avx512f::Load<align>(src + stride + 0));
            __m512 hi = _mm512_max_ps(Avx512f::Load<align>(src + F), Avx512f::Load<align>(src + stride + F));
            __m512 _lo = _mm512_shuffle_f32x4(lo, hi, 0x88);
            __m512 _hi = _mm512_shuffle_f32x4(lo, hi, 0xDD);
            return _mm512_max_ps(_mm512_shuffle_ps(_lo, _hi, 0x88), _mm512_shuffle_ps(_lo, _hi, 0xDD));
        }

        template <bool align> SIMD_INLINE __m512 Pooling2x2Max2(const float* src)
        {
            __m512 lo = Avx512f::Load<align>(src + 0);
            __m512 hi = Avx512f::Load<align>(src + F);
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
                    Avx512f::Store<align>(dst + (col >> 1), Pooling2x2Max2x2<align>(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Avx512f::Store<false>(dst + (col >> 1), Pooling2x2Max2x2<false>(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Simd::Max(src[widthEven], src[widthEven + srcStride]);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Avx512f::Store<align>(dst + (col >> 1), Pooling2x2Max2<align>(src + col));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Avx512f::Store<false>(dst + (col >> 1), Pooling2x2Max2<false>(src + col));
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
            return _mm512_max_ps(_mm512_max_ps(Avx512f::Load<align>(src), Avx512f::Load<align>(src + stride)), Avx512f::Load<align>(src + 2 * stride));
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
            return _mm512_max_ps(Avx512f::Load<align>(src), Avx512f::Load<align>(src + stride));
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
                    Avx512f::Store<false, true>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride), __mmask16(0x7FFF));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - step;
                    Avx512f::Store<false, true>(dst + (col >> 1), Pooling2x2Max3x3<false>(src + col, srcStride), __mmask16(0x7FFF));
                }
                if (width - widthEven)
                    Sse2::Max2x3s(src + widthEven, srcStride, dst + (widthEven >> 1));
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += step)
                    Avx512f::Store<false, true>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride), __mmask16(0x7FFF));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - step;
                    Avx512f::Store<false, true>(dst + (col >> 1), Pooling2x2Max3x2<false>(src + col, srcStride), __mmask16(0x7FFF));
                }
                if (width - widthEven)
                    Sse2::Max2x2s(src + widthEven, srcStride, dst + (widthEven >> 1));
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

        template <bool align, bool mask> SIMD_INLINE void NeuralProductSum(const float* a, const float* b, size_t offset, __m512& sum, __mmask16 m = -1)
        {
            __m512 _a = Avx512f::Load<align, mask>(a + offset, m);
            __m512 _b = Avx512f::Load<align, mask>(b + offset, m);
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
            *sum = Avx512f::ExtractSum(sum0);
        }

        void NeuralProductSum(const float* a, const float* b, size_t size, float* sum)
        {
            if (Aligned(a) && Aligned(b))
                NeuralProductSum<true>(a, b, size, sum);
            else
                NeuralProductSum<false>(a, b, size, sum);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

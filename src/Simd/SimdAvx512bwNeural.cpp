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

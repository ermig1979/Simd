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
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdStream.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float* delta, const __m128& norm, const __m128& alpha, const __m128& epsilon, float* gradient, float* weight)
        {
            __m128 d = _mm_mul_ps(Sse::Load<align>(delta), norm);
            __m128 _gradient = _mm_add_ps(Sse::Load<align>(gradient), _mm_mul_ps(d, d));
            Sse::Store<align>(gradient, _gradient);
            Sse::Store<align>(weight, _mm_sub_ps(Sse::Load<align>(weight), _mm_mul_ps(_mm_mul_ps(alpha, d), _mm_rsqrt_ps(_mm_add_ps(_gradient, epsilon)))));
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

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, const __m128& value, float* dst)
        {
            Sse::Store<align>(dst, _mm_add_ps(Sse::Load<align>(dst), _mm_mul_ps(value, Sse::Load<align>(src))));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float* src, size_t aligned, size_t partial, size_t full, float value, float* dst)
        {
            size_t i = 0;
            if (partial)
            {
                __m128 _value = _mm_set1_ps(value);
                for (; i < aligned; i += QF)
                {
                    AddMultiplied<align>(src + i + F * 0, _value, dst + i + F * 0);
                    AddMultiplied<align>(src + i + F * 1, _value, dst + i + F * 1);
                    AddMultiplied<align>(src + i + F * 2, _value, dst + i + F * 2);
                    AddMultiplied<align>(src + i + F * 3, _value, dst + i + F * 3);
                }
                for (; i < partial; i += F)
                    AddMultiplied<align>(src + i, _value, dst + i);
            }
            for (; i < full; ++i)
                dst[i] += src[i] * value;
        }

        void NeuralAddVectorMultipliedByValue(const float* src, size_t size, const float* value, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddMultiplied<true>(src, aligned, partial, size, *value, dst);
            else
                AddMultiplied<false>(src, aligned, partial, size, *value, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AddVector(const float* src, float* dst)
        {
            Sse::Store<align>(dst, _mm_add_ps(Sse::Load<align>(dst), Sse::Load<align>(src)));
        }

        template <bool align> SIMD_INLINE void AddVector(const float* src, size_t aligned, size_t partial, size_t full, float* dst)
        {
            size_t i = 0;
            for (; i < aligned; i += QF)
            {
                AddVector<align>(src + i + F * 0, dst + i + F * 0);
                AddVector<align>(src + i + F * 1, dst + i + F * 1);
                AddVector<align>(src + i + F * 2, dst + i + F * 2);
                AddVector<align>(src + i + F * 3, dst + i + F * 3);
            }
            for (; i < partial; i += F)
                AddVector<align>(src + i, dst + i);
            for (; i < full; ++i)
                dst[i] += src[i];
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

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void AddValue(const __m128& value, float* dst)
        {
            Sse::Store<align>(dst, _mm_add_ps(Sse::Load<align>(dst), value));
        }

        template <bool align> SIMD_INLINE void AddValue(const float* value, float* dst, size_t aligned, size_t partial, size_t full)
        {
            size_t i = 0;
            if (partial)
            {
                __m128 _value = _mm_set1_ps(value[0]);
                for (; i < aligned; i += QF)
                {
                    AddValue<align>(_value, dst + i + F * 0);
                    AddValue<align>(_value, dst + i + F * 1);
                    AddValue<align>(_value, dst + i + F * 2);
                    AddValue<align>(_value, dst + i + F * 3);
                }
                for (; i < partial; i += F)
                    AddValue<align>(_value, dst + i);
            }
            for (; i < full; ++i)
                dst[i] += value[0];
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

        //---------------------------------------------------------------------

        template <bool inversion> __m128i Invert(__m128i value);

        template <> __m128i Invert<true>(__m128i value)
        {
            return _mm_sub_epi8(K_INV_ZERO, value);
        }

        template <> __m128i Invert<false>(__m128i value)
        {
            return value;
        }

        template <bool align, bool stream> void Convert(__m128i src, const __m128 &_1_255, float * dst)
        {
            Sse::Stream<align, stream>(dst + 0, _mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(src)), _1_255));
            Sse::Stream<align, stream>(dst + 4, _mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(src)), _1_255));
        }

        template <bool inversion, bool align, bool stream> void Convert(const uint8_t * src, const __m128 &_1_255, float * dst)
        {
            __m128i _src = Invert<inversion>(Load<align>((__m128i*)src));
            Convert<align, stream>(UnpackU8<0>(_src), _1_255, dst + 0);
            Convert<align, stream>(UnpackU8<1>(_src), _1_255, dst + 8);
        }

        template <bool inversion, bool align, bool stream> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = AlignLo(width, A);
            __m128 _1_255 = _mm_set1_ps(1.0f / 255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    Convert<inversion, align, stream>(src + col, _1_255, dst + col);
                if (width != alignedWidth)
                    Convert<inversion, false, stream>(src + width - A, _1_255, dst + width - A);
                src += srcStride;
                dst += dstStride;
            }
            if (stream)
                _mm_mfence();
        }

        template <bool inversion> void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
            {
                if (width*height * sizeof(float) >= STREAM_SIZE_MIN)
                    NeuralConvert<inversion, true, true>(src, srcStride, width, height, dst, dstStride);
                else
                    NeuralConvert<inversion, true, false>(src, srcStride, width, height, dst, dstStride);
            }
            else
                NeuralConvert<inversion, false, false>(src, srcStride, width, height, dst, dstStride);
        }

        void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion)
        {
            if (inversion)
                NeuralConvert<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralConvert<false>(src, srcStride, width, height, dst, dstStride);
        }

        //---------------------------------------------------------------------

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
                __m128 mask = _mm_cmpgt_ps(Sse::Load<align>(src + i), _0);
                __m128 _dst = Sse::Load<align>(dst + i);
                Sse::Store<align>(dst + i, _mm_mul_ps(_mm_add_ps(_s, _mm_and_ps(mask, d)), _dst));
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

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralDerivativeSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(*slope);
            __m128 _1 = _mm_set1_ps(1.0f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 _src = Sse::Load<align>(src + i);
                __m128 _dst = Sse::Load<align>(dst + i);
                Sse::Store<align>(dst + i, _mm_mul_ps(_mm_mul_ps(_dst, _slope), _mm_mul_ps(_mm_sub_ps(_1, _src), _src)));
            }
            for (; i < size; ++i)
                dst[i] *= slope[0] * Base::DerivativeSigmoid(src[i]);
        }

        void NeuralDerivativeSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeSigmoid<true>(src, size, slope, dst);
            else
                NeuralDerivativeSigmoid<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

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
                __m128 _src = Sse::Load<align>(src + i);
                __m128 _dst = Sse::Load<align>(dst + i);
                Sse::Store<align>(dst + i, _mm_mul_ps(_mm_mul_ps(_dst, _slope), _mm_sub_ps(_1, _mm_mul_ps(_src, _src))));
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

        //---------------------------------------------------------------------

        template<bool align> void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            float e = exponent[0];
            size_t alignedSize = AlignLo(size, F);
            __m128 _e = _mm_set1_ps(e);
            Pow pow;
            size_t i = 0;
            for (; i < alignedSize; i += F)
                Sse::Store<align>(dst + i, pow(Sse::Load<align>(src + i), _e));
            for (; i < size; ++i)
                dst[i] = Base::Pow(src[i], e);
        }

        void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralPow<true>(src, size, exponent, dst);
            else
                NeuralPow<false>(src, size, exponent, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralProductSum(const float* a, const float* b, size_t offset, __m128& sum)
        {
            __m128 _a = Sse::Load<align>(a + offset);
            __m128 _b = Sse::Load<align>(b + offset);
            sum = _mm_add_ps(sum, _mm_mul_ps(_a, _b));
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float* a, const float* b, size_t size, float* sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        NeuralProductSum<align>(a, b, i + F * 0, sums[0]);
                        NeuralProductSum<align>(a, b, i + F * 1, sums[1]);
                        NeuralProductSum<align>(a, b, i + F * 2, sums[2]);
                        NeuralProductSum<align>(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += F)
                    NeuralProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                *sum += a[i] * b[i];
        }

        void NeuralProductSum(const float* a, const float* b, size_t size, float* sum)
        {
            if (Aligned(a) && Aligned(b))
                NeuralProductSum<true>(a, b, size, sum);
            else
                NeuralProductSum<false>(a, b, size, sum);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(*slope);
            __m128 _0 = _mm_set1_ps(-0.0f);
            __m128 _1 = _mm_set1_ps(1.0f);
            __m128 _a = _mm_set1_ps(0.5417f);
            __m128 _b = _mm_set1_ps(0.1460f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 _src = Sse::Load<align>(src + i);
                __m128 x = _mm_andnot_ps(_0, _mm_mul_ps(_src, _slope));
                __m128 x2 = _mm_mul_ps(x, x);
                __m128 x4 = _mm_mul_ps(x2, x2);
                __m128 series = _mm_add_ps(_mm_add_ps(_1, x), _mm_add_ps(_mm_mul_ps(x2, _a), _mm_mul_ps(x4, _b)));
                __m128 mask = _mm_cmpgt_ps(_src, _0);
                __m128 exp = _mm_or_ps(_mm_and_ps(_mm_rcp_ps(series), mask), _mm_andnot_ps(mask, series));
                __m128 sigmoid = _mm_rcp_ps(_mm_add_ps(_1, exp));
                Sse::Store<align>(dst + i, sigmoid);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
        }

        void NeuralRoughSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float* src, const __m128& k, const __m128& o, const __m128& m, float* dst)
        {
            __m128 _src = Sse::Load<align>(src);
            __m128 e1 = _mm_max_ps(m, _mm_sub_ps(o, _mm_mul_ps(_src, k)));
            __m128 e2 = _mm_mul_ps(e1, e1);
            __m128 e4 = _mm_mul_ps(e2, e2);
            __m128 e8 = _mm_mul_ps(e4, e4);
            __m128 e16 = _mm_mul_ps(e8, e8);
            __m128 e32 = _mm_mul_ps(e16, e16);
            __m128 e64 = _mm_mul_ps(e32, e32);
            __m128 sigmoid = _mm_rcp_ps(_mm_add_ps(o, _mm_mul_ps(e64, e64)));
            Sse::Store<align>(dst, sigmoid);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            __m128 _k = _mm_set1_ps((*slope) * 0.0078125f);
            __m128 _1 = _mm_set1_ps(1.0f);
            __m128 _05 = _mm_set1_ps(0.5f);
            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
            {
                NeuralRoughSigmoid2<align>(src + i + 0 * F, _k, _1, _05, dst + i + 0 * F);
                NeuralRoughSigmoid2<align>(src + i + 1 * F, _k, _1, _05, dst + i + 1 * F);
                NeuralRoughSigmoid2<align>(src + i + 2 * F, _k, _1, _05, dst + i + 2 * F);
                NeuralRoughSigmoid2<align>(src + i + 3 * F, _k, _1, _05, dst + i + 3 * F);
            }
            for (; i < partialAlignedSize; i += F)
                NeuralRoughSigmoid2<align>(src + i, _k, _1, _05, dst + i);
            for (; i < size; ++i)
                dst[i] = Base::RoughSigmoid2(src[i] * slope[0]);
        }

        void NeuralRoughSigmoid2(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid2<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid2<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void NeuralRoughTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t alignedSize = Simd::AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(*slope);
            __m128 _0 = _mm_set1_ps(-0.0f);
            __m128 _1 = _mm_set1_ps(1.0f);
            __m128 _a = _mm_set1_ps(0.5658f);
            __m128 _b = _mm_set1_ps(0.1430f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m128 _src = Sse::Load<align>(src + i);
                __m128 x = _mm_andnot_ps(_0, _mm_mul_ps(_src, _slope));
                __m128 x2 = _mm_mul_ps(x, x);
                __m128 x4 = _mm_mul_ps(x2, x2);
                __m128 pe = _mm_add_ps(_mm_add_ps(_1, x), _mm_add_ps(_mm_mul_ps(x2, _a), _mm_mul_ps(x4, _b)));
                __m128 ne = _mm_rcp_ps(pe);
                __m128 absTanh = _mm_mul_ps(_mm_sub_ps(pe, ne), _mm_rcp_ps(_mm_add_ps(pe, ne)));
                __m128 tanh = _mm_xor_ps(absTanh, _mm_and_ps(_0, _mm_cmpgt_ps(_0, _src)));
                Sse::Store<align>(dst + i, tanh);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughTanh(src[i] * slope[0]);
        }

        void NeuralRoughTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughTanh<true>(src, size, slope, dst);
            else
                NeuralRoughTanh<false>(src, size, slope, dst);
        }
    }
#endif// SIMD_SSE2_ENABLE
}

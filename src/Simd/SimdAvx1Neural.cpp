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
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t offset, __m256 & sum)
        {
            __m256 _a = Load<align>(a + offset);
            __m256 _b = Load<align>(b + offset);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_a, _b));
        }

        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if(partialAlignedSize)
            {
                __m256 sums[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
                if(fullAlignedSize)
                {
                    for(; i < fullAlignedSize; i += QF)
                    {
						NeuralProductSum<align>(a, b, i + F*0, sums[0]);
						NeuralProductSum<align>(a, b, i + F*1, sums[1]);
						NeuralProductSum<align>(a, b, i + F*2, sums[2]);
						NeuralProductSum<align>(a, b, i + F*3, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
                }
                for(; i < partialAlignedSize; i += F)
					NeuralProductSum<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for(; i < size; ++i)
                *sum += a[i]*b[i];
        }

        void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            if(Aligned(a) && Aligned(b))
				NeuralProductSum<true>(a, b, size, sum);
            else
				NeuralProductSum<false>(a, b, size, sum);
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, const __m256 & value, float * dst)
        {
            Store<align>(dst, _mm256_add_ps(Load<align>(dst), _mm256_mul_ps(value, Load<align>(src))));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
        {
            size_t i = 0;
            if (partial)
            {
                __m256 _value = _mm256_set1_ps(value);
                for (; i < aligned; i += QF)
                {
                    AddMultiplied<align>(src + i + F*0, _value, dst + i + F*0);
                    AddMultiplied<align>(src + i + F*1, _value, dst + i + F*1);
                    AddMultiplied<align>(src + i + F*2, _value, dst + i + F*2);
                    AddMultiplied<align>(src + i + F*3, _value, dst + i + F*3);
                }
                for (; i < partial; i += F)
                    AddMultiplied<align>(src + i, _value, dst + i);
            }
            for (; i < full; ++i)
                dst[i] += src[i] * value;
        }

        void NeuralAddVectorMultipliedByValue(const float * src, size_t size, const float * value, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddMultiplied<true>(src, aligned, partial, size, *value, dst);
            else
                AddMultiplied<false>(src, aligned, partial, size, *value, dst);
        }

        template <bool align> SIMD_INLINE void AddVector(const float * src, float * dst)
        {
            Store<align>(dst, _mm256_add_ps(Load<align>(dst), Load<align>(src)));
        }

        template <bool align> SIMD_INLINE void AddVector(const float * src, size_t aligned, size_t partial, size_t full, float * dst)
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

        void NeuralAddVector(const float * src, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(src) && Aligned(dst))
                AddVector<true>(src, aligned, partial, size, dst);
            else
                AddVector<false>(src, aligned, partial, size, dst);
        }

        template <bool align> SIMD_INLINE void AddValue(const __m256 & value, float * dst)
        {
            Store<align>(dst, _mm256_add_ps(Load<align>(dst), value));
        }

        template <bool align> SIMD_INLINE void AddValue(const float * value, float * dst, size_t aligned, size_t partial, size_t full)
        {
            size_t i = 0;
            if (partial)
            {
                __m256 _value = _mm256_set1_ps(value[0]);
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

        void NeuralAddValue(const float * value, float * dst, size_t size)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            if (Aligned(dst))
                AddValue<true>(value, dst, aligned, partial, size);
            else
                AddValue<false>(value, dst, aligned, partial, size);
        }

		template <bool align> SIMD_INLINE void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			size_t alignedSize = Simd::AlignLo(size, F);
			__m256 _slope = _mm256_set1_ps(*slope);
			__m256 _0 = _mm256_set1_ps(-0.0f);
			__m256 _1 = _mm256_set1_ps(1.0f);
			__m256 _a = _mm256_set1_ps(0.5417f);
			__m256 _b = _mm256_set1_ps(0.1460f);
			size_t i = 0;
			for (; i < alignedSize; i += F)
			{
				__m256 _src = Load<align>(src + i);
				__m256 x = _mm256_andnot_ps(_0, _mm256_mul_ps(_src, _slope));
				__m256 x2 = _mm256_mul_ps(x, x);
				__m256 x4 = _mm256_mul_ps(x2, x2);
				__m256 series = _mm256_add_ps(_mm256_add_ps(_1, x), _mm256_add_ps(_mm256_mul_ps(x2, _a), _mm256_mul_ps(x4, _b)));
				__m256 mask = _mm256_cmp_ps(_src, _0, _CMP_GT_OS);
                __m256 exp = _mm256_blendv_ps(series, _mm256_rcp_ps(series), mask);
                __m256 sigmoid = _mm256_rcp_ps(_mm256_add_ps(_1, exp));
				Store<align>(dst + i, sigmoid);
			}
			for (; i < size; ++i)
				dst[i] = Base::RoughSigmoid(src[i] * slope[0]);
		}

		void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			if (Aligned(src) && Aligned(dst))
				NeuralRoughSigmoid<true>(src, size, slope, dst);
			else
				NeuralRoughSigmoid<false>(src, size, slope, dst);
		}

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, const __m256 & k, const __m256 & o, const __m256 & m, float * dst)
        {
            __m256 _src = Load<align>(src);
            __m256 e1 = _mm256_max_ps(m, _mm256_sub_ps(o, _mm256_mul_ps(_src, k)));
            __m256 e2 = _mm256_mul_ps(e1, e1);
            __m256 e4 = _mm256_mul_ps(e2, e2);
            __m256 e8 = _mm256_mul_ps(e4, e4);
            __m256 e16 = _mm256_mul_ps(e8, e8);
            __m256 e32 = _mm256_mul_ps(e16, e16);
            __m256 e64 = _mm256_mul_ps(e32, e32);
            __m256 sigmoid = _mm256_rcp_ps(_mm256_add_ps(o, _mm256_mul_ps(e64, e64)));
            Store<align>(dst, sigmoid);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            __m256 _k = _mm256_set1_ps((*slope)*0.0078125f);
            __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 _05 = _mm256_set1_ps(0.5f);
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

        void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid2<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid2<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void NeuralDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, F);
            __m256 _slope = _mm256_set1_ps(*slope);
            __m256 _1 = _mm256_set1_ps(1.0f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m256 _src = Load<align>(src + i); 
                __m256 _dst = Load<align>(dst + i);
                Store<align>(dst + i, _mm256_mul_ps(_mm256_mul_ps(_dst, _slope), _mm256_mul_ps(_mm256_sub_ps(_1, _src), _src)));
            }
            for (; i < size; ++i)
                dst[i] *= slope[0] * Base::DerivativeSigmoid(src[i]);
        }

        void NeuralDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeSigmoid<true>(src, size, slope, dst);
            else
                NeuralDerivativeSigmoid<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void NeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, F);
            __m256 _slope = _mm256_set1_ps(*slope);
            __m256 _0 = _mm256_set1_ps(-0.0f);
            __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 _a = _mm256_set1_ps(0.5658f);
            __m256 _b = _mm256_set1_ps(0.1430f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m256 _src = Load<align>(src + i);
                __m256 x = _mm256_andnot_ps(_0, _mm256_mul_ps(_src, _slope));
                __m256 x2 = _mm256_mul_ps(x, x);
                __m256 x4 = _mm256_mul_ps(x2, x2);
                __m256 pe = _mm256_add_ps(_mm256_add_ps(_1, x), _mm256_add_ps(_mm256_mul_ps(x2, _a), _mm256_mul_ps(x4, _b)));
                __m256 ne = _mm256_rcp_ps(pe);
                __m256 absTanh = _mm256_mul_ps(_mm256_sub_ps(pe, ne), _mm256_rcp_ps(_mm256_add_ps(pe, ne)));
                __m256 tanh = _mm256_xor_ps(absTanh, _mm256_and_ps(_0, _mm256_cmp_ps(_0, _src, _CMP_GT_OS)));
                Store<align>(dst + i, tanh);
            }
            for (; i < size; ++i)
                dst[i] = Base::RoughTanh(src[i] * slope[0]);
        }

        void NeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughTanh<true>(src, size, slope, dst);
            else
                NeuralRoughTanh<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void NeuralDerivativeTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, F);
            __m256 _slope = _mm256_set1_ps(*slope);
            __m256 _1 = _mm256_set1_ps(1.0f);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m256 _src = Load<align>(src + i);
                __m256 _dst = Load<align>(dst + i);
                Store<align>(dst + i, _mm256_mul_ps(_mm256_mul_ps(_dst, _slope), _mm256_sub_ps(_1, _mm256_mul_ps(_src, _src))));
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

        template <bool align> void NeuralRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            assert(s >= 0.0f && s <= 1.0f);
            size_t alignedSize = Simd::AlignLo(size, F);
            size_t i = 0;
            if (s == 0)
            {
                __m256 _0 = _mm256_set1_ps(0.0f);
                for (; i < alignedSize; i += F)
                {
                    __m256 _src = Load<align>(src + i);
                    Store<align>(dst + i, _mm256_max_ps(_0, _src));
                }
                for (; i < size; ++i)
                    dst[i] = Simd::Max(0.0f, src[i]);
            }
            else
            {
                __m256 _s = _mm256_set1_ps(s);
                for (; i < alignedSize; i += F)
                {
                    __m256 _src = Load<align>(src + i);
                    Store<align>(dst + i, _mm256_max_ps(_mm256_mul_ps(_s, _src), _src));
                }
                for (; i < size; ++i)
                    dst[i] = Simd::Max(src[i] * s, src[i]);
            }
        }

        void NeuralRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRelu<true>(src, size, slope, dst);
            else
                NeuralRelu<false>(src, size, slope, dst);
        }

        template <bool align> void NeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            __m256 _0 = _mm256_set1_ps(0.0f);
            __m256 _1 = _mm256_set1_ps(1.0f);
            __m256 _s = _mm256_set1_ps(s);
            size_t alignedSize = Simd::AlignLo(size, F);
            size_t i = 0;
            for (; i < alignedSize; i += F)
            {
                __m256 mask = _mm256_cmp_ps(Load<align>(src + i), _0, _CMP_GT_OS);
                __m256 _dst = Load<align>(dst + i);
                Store<align>(dst + i, _mm256_mul_ps(_mm256_blendv_ps(_s, _1, mask), _dst));
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

        template <bool align> SIMD_INLINE void UpdateWeights(const float * x, const __m256 & a, const __m256 & b, float * d, float * w)
        {
            __m256 _d = _mm256_add_ps(_mm256_mul_ps(a, Load<align>(d)), _mm256_mul_ps(b, Load<align>(x)));
            Store<align>(d, _d);
            Store<align>(w, _mm256_add_ps(Load<align>(w), _d));
        }

        template <bool align> SIMD_INLINE void UpdateWeights(const float * x, size_t offset, const __m256 & a, const __m256 & b, float * d, float * w)
        {
            UpdateWeights<align>(x + offset, a, b, d + offset, w + offset);
        }

        template <bool align> SIMD_INLINE void NeuralUpdateWeights(const float * x, size_t size, const float & a, const float & b, float * d, float * w)
        {
            if (align)
                assert(Aligned(x) && Aligned(d) && Aligned(w));

            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            __m256 _a = _mm256_set1_ps(a);
            __m256 _b = _mm256_set1_ps(b);
            size_t i = 0;
            if (partialAlignedSize)
            {
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        UpdateWeights<align>(x, i + F*0, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F*1, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F*2, _a, _b, d, w);
                        UpdateWeights<align>(x, i + F*3, _a, _b, d, w);
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

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float * delta, const __m256 & norm, const __m256 & alpha, const __m256 & epsilon, float * gradient, float * weight)
        {
            __m256 d = _mm256_mul_ps(Load<align>(delta), norm);
            __m256 _gradient = _mm256_add_ps(Load<align>(gradient), _mm256_mul_ps(d, d));
            Store<align>(gradient, _gradient);
            Store<align>(weight, _mm256_sub_ps(Load<align>(weight), _mm256_mul_ps(_mm256_mul_ps(alpha, d), _mm256_rsqrt_ps(_mm256_add_ps(_gradient, epsilon)))));
        }

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float * delta, size_t offset, const __m256 & norm, const __m256 & alpha, const __m256 & epsilon, float * gradient, float * weight)
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

        void NeuralAdaptiveGradientUpdate(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight)
        {
            if (Aligned(delta) && Aligned(gradient) && Aligned(weight))
                NeuralAdaptiveGradientUpdate<true>(delta, size, batch, alpha, epsilon, gradient, weight);
            else
                NeuralAdaptiveGradientUpdate<false>(delta, size, batch, alpha, epsilon, gradient, weight);
        }

        template <size_t size> SIMD_INLINE void LoadWeightsForward(const float * src, __m256 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm256_set1_ps(src[i]);
        }

        template <size_t size> SIMD_INLINE void LoadWeightsBackward(const float * src, __m256 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm256_set1_ps(src[size - i - 1]);
        }

        namespace
        {
            template<int count> struct Buffer
            {
                Buffer(size_t width)
                {
                    _size = width*sizeof(float);
                    size_t stride = AlignHi(width + 2 * (count - 1), F);
                    size_t full = count*stride*sizeof(float);
                    _ptr = Allocate(full);
                    memset(_ptr, 0, full);
                    rows[0] = (float*)_ptr;
                    for (size_t i = 1; i < count; ++i)
                        rows[i] = rows[i - 1] + stride;
                }

                void Update(const float * src)
                {
                    float * tmp = rows[0];
                    if (src == NULL)
                        memset(tmp + count - 1, 0, _size);
                    else
                        memcpy(tmp + count - 1, src, _size);
                    for (size_t i = 0; i < count - 1; ++i)
                        rows[i] = rows[i + 1];
                    rows[count - 1] = tmp;
                }

                ~Buffer()
                {
                    Free(_ptr);
                }

                float * rows[count];
            private:
                size_t _size;
                void * _ptr;
            };
        }

        template<size_t coreX, size_t coreY> struct Convolution
        {
            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights);

            template<bool align> static SIMD_INLINE __m256 Backward(const Buffer<coreX> & buffer, size_t offset, const __m256 * weights);
 
            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums);
        };

        template<> struct Convolution<2, 2>
        {
            template <bool align> static SIMD_INLINE __m256 Convolution2(const float * src, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_mul_ps(Load<align>(src), weights[0]),
                    _mm256_mul_ps(Load<false>(src + 1), weights[1]));
            }
            
            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights)
            {
                return _mm256_add_ps(Convolution2<align>(src, weights),
                    Convolution2<align>(src + stride, weights + 2));
            }

            template<bool align> static SIMD_INLINE __m256 Backward(const Buffer<2> & buffer, size_t offset, const __m256 * weights)
            {
                return _mm256_add_ps(Convolution2<align>(buffer.rows[0] + offset, weights),
                    Convolution2<align>(buffer.rows[1] + offset, weights + 2));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m256 & dst, __m256 * sums)
            {
                sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(dst, Load<false>(src + 1)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 2);
            }
        };

        template<> struct Convolution<3, 3>
        {
            template <bool align> static SIMD_INLINE __m256 Convolution3(const float * src, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_mul_ps(Load<align>(src), weights[0]),
                    _mm256_add_ps(_mm256_mul_ps(Load<false>(src + 1), weights[1]),
                        _mm256_mul_ps(Load<false>(src + 2), weights[2])));
            }

            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights)
            {
                return _mm256_add_ps(Convolution3<align>(src, weights),
                    _mm256_add_ps(Convolution3<align>(src + stride, weights + 3),
                    Convolution3<align>(src + 2 * stride, weights + 6)));
            }

            template<bool align> static SIMD_INLINE __m256 Backward(const Buffer<3> & buffer, size_t offset, const __m256 * weights)
            {
                return _mm256_add_ps(Convolution3<align>(buffer.rows[0] + offset, weights),
                    _mm256_add_ps(Convolution3<align>(buffer.rows[1] + offset, weights + 3),
                    Convolution3<align>(buffer.rows[2] + offset, weights + 6)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m256 & dst, __m256 * sums)
            {
                sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(dst, Load<false>(src + 2)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 3);
                Sum<align>(src + stride * 2, dst, sums + 6);
            }
        };

        template<> struct Convolution<4, 4>
        {
            template <bool align> static SIMD_INLINE __m256 Convolution4(const float * src, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(Load<align>(src), weights[0]), _mm256_mul_ps(Load<false>(src + 1), weights[1])),
                    _mm256_add_ps(_mm256_mul_ps(Load<false>(src + 2), weights[2]), _mm256_mul_ps(Load<false>(src + 3), weights[3])));
            }

            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_add_ps(Convolution4<align>(src, weights), 
                    Convolution4<align>(src + stride, weights + 4)),
                    _mm256_add_ps(Convolution4<align>(src + 2 * stride, weights + 8), 
                    Convolution4<align>(src + 3 * stride, weights + 12)));
            }

            template<bool align> static SIMD_INLINE __m256 Backward(const Buffer<4> & buffer, size_t offset, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_add_ps(Convolution4<align>(buffer.rows[0] + offset, weights),
                    Convolution4<align>(buffer.rows[1] + offset, weights + 4)),
                    _mm256_add_ps(Convolution4<align>(buffer.rows[2] + offset, weights + 8),
                    Convolution4<align>(buffer.rows[3] + offset, weights + 12)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m256 & dst, __m256 * sums)
            {
                sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(dst, Load<false>(src + 2)));
                sums[3] = _mm256_add_ps(sums[3], _mm256_mul_ps(dst, Load<false>(src + 3)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 4);
                Sum<align>(src + stride * 2, dst, sums + 8);
                Sum<align>(src + stride * 3, dst, sums + 12);
            }
        };

        template<> struct Convolution<5, 5>
        {
            template <bool align> static SIMD_INLINE __m256 Convolution5(const float * src, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_mul_ps(Load<align>(src), weights[0]), _mm256_add_ps(
                    _mm256_add_ps(_mm256_mul_ps(Load<false>(src + 1), weights[1]), _mm256_mul_ps(Load<false>(src + 2), weights[2])),
                    _mm256_add_ps(_mm256_mul_ps(Load<false>(src + 3), weights[3]), _mm256_mul_ps(Load<false>(src + 4), weights[4]))));
            }

            template<bool align> static SIMD_INLINE __m256 Forward(const float * src, size_t stride, const __m256 * weights)
            {
                return _mm256_add_ps(Convolution5<align>(src, weights), 
                    _mm256_add_ps(_mm256_add_ps(Convolution5<align>(src + stride, weights + 5), 
                    Convolution5<align>(src + 2 * stride, weights + 10)),
                    _mm256_add_ps(Convolution5<align>(src + 3 * stride, weights + 15), 
                    Convolution5<align>(src + 4 * stride, weights + 20))));
            }

            template<bool align> static SIMD_INLINE __m256 Backward(const Buffer<5> & buffer, size_t offset, const __m256 * weights)
            {
                return _mm256_add_ps(_mm256_add_ps(Convolution5<align>(buffer.rows[0] + offset, weights),
                    _mm256_add_ps(Convolution5<align>(buffer.rows[1] + offset, weights + 5),
                    Convolution5<align>(buffer.rows[2] + offset, weights + 10))),
                    _mm256_add_ps(Convolution5<align>(buffer.rows[3] + offset, weights + 15),
                    Convolution5<align>(buffer.rows[4] + offset, weights + 20)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, const __m256 & dst, __m256 * sums)
            {
                sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(dst, Load<align>(src + 0)));
                sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(dst, Load<false>(src + 1)));
                sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(dst, Load<false>(src + 2)));
                sums[3] = _mm256_add_ps(sums[3], _mm256_mul_ps(dst, Load<false>(src + 3)));
                sums[4] = _mm256_add_ps(sums[4], _mm256_mul_ps(dst, Load<false>(src + 4)));
            }

            template <bool align> static SIMD_INLINE void Sum(const float * src, size_t stride, const __m256 & dst, __m256 * sums)
            {
                Sum<align>(src + stride * 0, dst, sums + 0);
                Sum<align>(src + stride * 1, dst, sums + 5);
                Sum<align>(src + stride * 2, dst, sums + 10);
                Sum<align>(src + stride * 3, dst, sums + 15);
                Sum<align>(src + stride * 4, dst, sums + 20);
            }
        };

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionForward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, F);
            __m256 tailMask = RightNotZero(width - alignedWidth);
            __m256 _weights[coreX*coreY];
            LoadWeightsForward<coreX*coreY>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += F)
                {
                    __m256 _dst = Load<align>(dst + col);
                    _dst = _mm256_add_ps(_dst, Convolution<coreX, coreY>::template Forward<align>(src + col, srcStride, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m256 _dst = Load<false>(dst + col);
                    _dst = _mm256_add_ps(_dst, _mm256_and_ps(tailMask, Convolution<coreX, coreY>::template Forward<false>(src + col, srcStride, _weights)));
                    Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution2x2Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionForward<true, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionForward<false, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template<bool condition> struct If
        {
            template<bool align> static SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
            {
                Avx::AddMultiplied<align>(src, aligned, partial, full, value, dst);
            }
        };

        template<> struct If<false>
        {
            template<bool align> static SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
            {
            }
        };

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackwardSmall(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t aligned = AlignLo(width, QF);
            size_t partial = AlignLo(width, F);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < coreY; ++dy)
                {
                    const float * w = weights + dy * coreX;
                    float * d = dst + dy*dstStride;
                    If<0 < coreX>::template AddMultiplied<align>(src, aligned, partial, width, w[0], d + 0);
                    If<1 < coreX>::template AddMultiplied<false>(src, aligned, partial, width, w[1], d + 1);
                    If<2 < coreX>::template AddMultiplied<false>(src, aligned, partial, width, w[2], d + 2);
                    If<3 < coreX>::template AddMultiplied<false>(src, aligned, partial, width, w[3], d + 3);
                    If<4 < coreX>::template AddMultiplied<false>(src, aligned, partial, width, w[4], d + 4);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackwardLarge(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            Buffer<coreX> buffer(width);
            height += coreY - 1;
            width += coreX - 1;
            size_t alignedWidth = AlignLo(width, F);
            __m256 tailMask = RightNotZero(width - alignedWidth);
            __m256 _weights[coreX*coreY];
            LoadWeightsBackward<coreX*coreY>(weights, _weights);

            for (size_t row = 0; row < height; ++row)
            {
                buffer.Update(row <= height - coreY ? src : NULL);
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m256 _dst = Load<align>(dst + col);
                    _dst = _mm256_add_ps(_dst, Convolution<coreX, coreY>::template Backward<true>(buffer, col, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m256 _dst = Load<false>(dst + col);
                    _dst = _mm256_add_ps(_dst, _mm256_and_ps(tailMask, Convolution<coreX, coreY>::template Backward<false>(buffer, col, _weights)));
                    Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align, size_t coreX, size_t coreY> void NeuralAddConvolutionBackward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (width*height < 1024)
                NeuralAddConvolutionBackwardSmall<align, coreX, coreY>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackwardLarge<align, coreX, coreY>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution2x2Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionBackward<true, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolutionBackward<false, 5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align, size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionSum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m256 tailMask = RightNotZero(width - alignedWidth);
            __m256 _sums[coreX*coreY];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m256 _dst = Load<align>(dst + col);
                    Convolution<coreX, coreY>::template Sum<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m256 _dst = _mm256_and_ps(tailMask, Load<false>(dst + col));
                    Convolution<coreX, coreY>::template Sum<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            size_t i = 0, n = Simd::AlignLo(coreX*coreY, F);
            for (; i < n; i += F)
                Add8ExtractedSums(_sums + i, sums + i);
            for (; i < coreX*coreY; ++i)
                sums[i] += ExtractSum(_sums[i]);
        }

        void NeuralAddConvolution2x2Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 2, 2>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 3, 3>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution4x4Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 4, 4>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolutionSum<true, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolutionSum<false, 5, 5>(src, srcStride, dst, dstStride, width, height, sums);
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max2x2(const float * src, size_t stride)
        {
            __m256 lo = _mm256_max_ps(Load<align>(src + 0), Load<align>(src + stride + 0));
            __m256 hi = _mm256_max_ps(Load<align>(src + F), Load<align>(src + stride + F));
            __m256 _lo = _mm256_permute2f128_ps(lo, hi, 0x20);
            __m256 _hi = _mm256_permute2f128_ps(lo, hi, 0x31);
            return _mm256_max_ps(_mm256_shuffle_ps(_lo, _hi, 0x88), _mm256_shuffle_ps(_lo, _hi, 0xDD));
        }

        template <bool align> SIMD_INLINE __m256 Pooling2x2Max2(const float * src)
        {
            __m256 lo = Load<align>(src + 0);
            __m256 hi = Load<align>(src + F);
            __m256 _lo = _mm256_permute2f128_ps(lo, hi, 0x20);
            __m256 _hi = _mm256_permute2f128_ps(lo, hi, 0x31);
            return _mm256_max_ps(_mm256_shuffle_ps(_lo, _hi, 0x88), _mm256_shuffle_ps(_lo, _hi, 0xDD));
        }

        template <bool align> void NeuralPooling2x2Max2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
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
                src += 2*srcStride;
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

        template <bool align> static SIMD_INLINE void AddProductSum1x4x8(const __m256 & a, size_t K, const float * b, __m256 * sums)
        {
            sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(a, Load<align>(b + 0 * K)));
            sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(a, Load<align>(b + 1 * K)));
            sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(a, Load<align>(b + 2 * K)));
            sums[3] = _mm256_add_ps(sums[3], _mm256_mul_ps(a, Load<align>(b + 3 * K)));
        }

        template <bool align> static SIMD_INLINE void AddProductSum1x1x8(const __m256 & a, const float * b, __m256 & sum)
        {
            sum = _mm256_add_ps(sum, _mm256_mul_ps(a, Load<align>(b)));
        }

        SIMD_INLINE void Add4ExtractedSums(const __m256 * src, float * dst)
        {
            __m256 sum256 = _mm256_hadd_ps(_mm256_hadd_ps(src[0], src[1]), _mm256_hadd_ps(src[2], src[3]));
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));
            _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), sum128));
        }

        template <bool align> static SIMD_INLINE void AddProductSum2x4x8(const __m256 & a0, const __m256 & a1, size_t K, const float * b, __m256 * sums)
        {
            __m256 b0 = Avx::Load<align>(b + 0 * K);
            sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(a0, b0));
            sums[4] = _mm256_add_ps(sums[4], _mm256_mul_ps(a1, b0));
            __m256 b1 = Avx::Load<align>(b + 1 * K);
            sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(a0, b1));
            sums[5] = _mm256_add_ps(sums[5], _mm256_mul_ps(a1, b1));
            __m256 b2 = Avx::Load<align>(b + 2 * K);
            sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(a0, b2));
            sums[6] = _mm256_add_ps(sums[6], _mm256_mul_ps(a1, b2));
            __m256 b3 = Avx::Load<align>(b + 3 * K);
            sums[3] = _mm256_add_ps(sums[3], _mm256_mul_ps(a0, b3));
            sums[7] = _mm256_add_ps(sums[7], _mm256_mul_ps(a1, b3));
        }

        template <bool align> static SIMD_INLINE void AddProductSum2x1x8(const __m256 & a0, const __m256 & a1, const float * b, __m256 * sums)
        {
            sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(a0, Load<align>(b)));
            sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(a1, Load<align>(b)));
        }

        template <bool align> void NeuralConvolutionForwardGemmNT(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
        {
            size_t M2 = Simd::AlignLo(M, 2);
            size_t N4 = Simd::AlignLo(N, 4);
            size_t K8 = Simd::AlignLo(K, 8);
            __m256 tailMask = RightNotZero(K - K8);
            size_t i = 0;
            for (; i < M2; i += 2)
            {
                const float * pa0 = a + i*K;
                const float * pa1 = a + i*K + K;
                float * pc0 = c + i*N;
                float * pc1 = c + i*N + N;
                size_t j = 0;
                for (; j < N4; j += 4)
                {
                    const float * pb = b + j*K;
                    __m256 sums[8] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                    for (size_t k = 0; k < K8; k += 8)
                    {
                        __m256 _a0 = Avx::Load<false>(pa0 + k);
                        __m256 _a1 = Avx::Load<false>(pa1 + k);
                        AddProductSum2x4x8<align>(_a0, _a1, K, pb + k, sums);
                    }
                    if (K8 < K)
                    {
                        size_t k = K - 8;
                        __m256 _a0 = _mm256_and_ps(tailMask, Avx::Load<false>(pa0 + k));
                        __m256 _a1 = _mm256_and_ps(tailMask, Avx::Load<false>(pa1 + k));
                        AddProductSum2x4x8<false>(_a0, _a1, K, pb + k, sums);
                    }
                    Add4ExtractedSums(sums + 0, pc0 + j);
                    Add4ExtractedSums(sums + 4, pc1 + j);
                }
                for (; j < N; ++j)
                {
                    const float * pb = b + j*K;
                    __m256 sums[2] = { _mm256_setzero_ps(), _mm256_setzero_ps() };
                    for (size_t k = 0; k < K8; k += 8)
                    {
                        __m256 _a0 = Avx::Load<false>(pa0 + k);
                        __m256 _a1 = Avx::Load<false>(pa1 + k);
                        AddProductSum2x1x8<align>(_a0, _a1, pb + k, sums);
                    }
                    if (K8 < K)
                    {
                        size_t k = K - 8;
                        __m256 _a0 = _mm256_and_ps(tailMask, Avx::Load<false>(pa0 + k));
                        __m256 _a1 = _mm256_and_ps(tailMask, Avx::Load<false>(pa1 + k));
                        AddProductSum2x1x8<false>(_a0, _a1, pb + k, sums);
                    }
                    pc0[j] += Avx::ExtractSum(sums[0]);
                    pc1[j] += Avx::ExtractSum(sums[1]);
                }
            }
            for (; i < M; ++i)
            {
                const float * pa = a + i*K;
                float * pc = c + i*N;
                size_t j = 0;
                for (; j < N4; j += 4)
                {
                    const float * pb = b + j*K;
                    __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                    for (size_t k = 0; k < K8; k += 8)
                    {
                        __m256 _a = Avx::Load<false>(pa + k);
                        AddProductSum1x4x8<align>(_a, K, pb + k, sums);
                    }
                    if (K8 < K)
                    {
                        size_t k = K - 8;
                        __m256 _a = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k));
                        AddProductSum1x4x8<false>(_a, K, pb + k, sums);
                    }
                    Add4ExtractedSums(sums + 0, pc + j);
                }
                for (; j < N; ++j)
                {
                    const float * pb = b + j*K;
                    __m256 sum = _mm256_setzero_ps();
                    for (size_t k = 0; k < K8; k += 8)
                    {
                        __m256 _a = Avx::Load<false>(pa + k);
                        AddProductSum1x1x8<align>(_a, pb + k, sum);
                    }
                    if (K8 < K)
                    {
                        size_t k = K - 8;
                        __m256 _a = _mm256_and_ps(tailMask, Avx::Load<false>(pa + k));
                        AddProductSum1x1x8<false>(_a, pb + k, sum);
                    }
                    pc[j] += Avx::ExtractSum(sum);
                }
            }
        }

        void NeuralConvolutionForward(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
            const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
            void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add)
        {
            assert(dstWidth == (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1);
            assert(dstHeight == (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1);

            if (!add)
                memset(dst, 0, dstWidth*dstHeight*dstDepth*sizeof(float));

            float * temporal = NULL;
            void * internal = NULL;

            bool transpose = dstWidth*dstHeight <= 1024;

            if (kernelX == 1 && kernelY == 1 && !transpose)
                temporal = (float*)src;
            else
            {
                size_t required = dstWidth*dstHeight*srcDepth*kernelX*kernelY*sizeof(float);
                if (buffer != AlignHi(buffer, SIMD_ALIGN))
                    required += SIMD_ALIGN;
                if (buffer == NULL || size == NULL || *size < required)
                {
                    internal = Allocate(required);
                    if (size)
                        *size = required;
                    temporal = (float*)internal;
                }
                else
                    temporal = (float*)AlignHi(buffer, SIMD_ALIGN);

                if (transpose)
                    Base::NeuralConvolutionForwardConvertT(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, temporal);
                else
                    Base::NeuralConvolutionForwardConvertN(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, temporal);
            }

            size_t M = dstDepth, N = dstHeight*dstWidth, K = kernelX*kernelY*srcDepth;
            if (transpose)
            {
                if (Aligned(K, F))
                    NeuralConvolutionForwardGemmNT<true>(M, N, K, weight, temporal, dst);
                else
                    NeuralConvolutionForwardGemmNT<false>(M, N, K, weight, temporal, dst);
            }
            else
                Base::NeuralConvolutionForwardGemmNN(M, N, K, weight, temporal, dst);

            if (internal)
                Free(internal);
        }
    }
#endif// SIMD_AVX_ENABLE
}

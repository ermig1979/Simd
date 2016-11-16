/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        template <bool align> SIMD_INLINE void NeuralProductSum(const float * a, const float * b, size_t offset, __m128 & sum)
        {
            __m128 _a = Load<align>(a + offset);
            __m128 _b = Load<align>(b + offset);
            sum = _mm_add_ps(sum, _mm_mul_ps(_a, _b));
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
                __m128 sums[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
                if(fullAlignedSize)
                {
                    for(; i < fullAlignedSize; i += QF)
                    {
						NeuralProductSum<align>(a, b, i + F*0, sums[0]);
						NeuralProductSum<align>(a, b, i + F*1, sums[1]);
						NeuralProductSum<align>(a, b, i + F*2, sums[2]);
						NeuralProductSum<align>(a, b, i + F*3, sums[3]);
                    }
                    sums[0] = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
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

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, const __m128 & value, float * dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), _mm_mul_ps(value, Load<align>(src))));
        }

        template <bool align> SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t partial, size_t full, float value, float * dst)
        {
            size_t i = 0;
            if (partial)
            {
                __m128 _value = _mm_set1_ps(value);
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

		template <bool align> SIMD_INLINE void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
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
				__m128 _src = Load<align>(src + i);
				__m128 x = _mm_andnot_ps(_0, _mm_mul_ps(_src, _slope));
				__m128 x2 = _mm_mul_ps(x, x);
				__m128 x4 = _mm_mul_ps(x2, x2);
				__m128 series = _mm_add_ps(_mm_add_ps(_1, x), _mm_add_ps(_mm_mul_ps(x2, _a), _mm_mul_ps(x4, _b)));
				__m128 mask = _mm_cmpgt_ps(_src, _0);
				__m128 exp = _mm_or_ps(_mm_and_ps(_mm_rcp_ps(series), mask), _mm_andnot_ps(mask, series));
				__m128 sigmoid = _mm_rcp_ps(_mm_add_ps(_1, exp));
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

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, const __m128 & k, const __m128 & o, const __m128 & m, float * dst)
        {
            __m128 _src = Load<align>(src);
            __m128 e1 = _mm_max_ps(m, _mm_sub_ps(o, _mm_mul_ps(_src, k)));
            __m128 e2 = _mm_mul_ps(e1, e1);
            __m128 e4 = _mm_mul_ps(e2, e2);
            __m128 e8 = _mm_mul_ps(e4, e4);
            __m128 e16 = _mm_mul_ps(e8, e8);
            __m128 e32 = _mm_mul_ps(e16, e16);
            __m128 e64 = _mm_mul_ps(e32, e32);
            __m128 sigmoid = _mm_rcp_ps(_mm_add_ps(o, _mm_mul_ps(e64, e64)));
            Store<align>(dst, sigmoid);
        }

        template <bool align> SIMD_INLINE void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            size_t partialAlignedSize = Simd::AlignLo(size, F);
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            __m128 _k = _mm_set1_ps((*slope)*0.0078125f);
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

        void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralRoughSigmoid2<true>(src, size, slope, dst);
            else
                NeuralRoughSigmoid2<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void NeuralDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst)
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
                Store<align>(dst + i, _mm_mul_ps(_mm_mul_ps(_dst, _slope), _mm_mul_ps(_mm_sub_ps(_1, _src), _src)));
            }
            for (; i < size; ++i)
                dst[i] *= slope[0]*Base::DerivativeSigmoid(src[i]);
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
                __m128 _src = Load<align>(src + i);
                __m128 x = _mm_andnot_ps(_0, _mm_mul_ps(_src, _slope));
                __m128 x2 = _mm_mul_ps(x, x);
                __m128 x4 = _mm_mul_ps(x2, x2);
                __m128 pe = _mm_add_ps(_mm_add_ps(_1, x), _mm_add_ps(_mm_mul_ps(x2, _a), _mm_mul_ps(x4, _b)));
                __m128 ne = _mm_rcp_ps(pe);
                __m128 absTanh = _mm_mul_ps(_mm_sub_ps(pe, ne), _mm_rcp_ps(_mm_add_ps(pe, ne)));
                __m128 tanh = _mm_xor_ps(absTanh, _mm_and_ps(_0, _mm_cmpgt_ps(_0, _src)));
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

        void NeuralDerivativeTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeTanh<true>(src, size, slope, dst);
            else
                NeuralDerivativeTanh<false>(src, size, slope, dst);
        }

        template <bool align> void NeuralRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float s = slope[0];
            assert(s >= 0.0f && s <= 1.0f);
            size_t alignedSize = Simd::AlignLo(size, F);
            size_t i = 0;
            if (s == 0)
            {
                __m128 _0 = _mm_set1_ps(0.0f);
                for (; i < alignedSize; i += F)
                {
                    __m128 _src = Load<align>(src + i);
                    Store<align>(dst + i, _mm_max_ps(_0, _src));
                }
                for (; i < size; ++i)
                    dst[i] = Simd::Max(0.0f, src[i]);
            }
            else
            {
                __m128 _s = _mm_set1_ps(s);
                for (; i < alignedSize; i += F)
                {
                    __m128 _src = Load<align>(src + i);
                    Store<align>(dst + i, _mm_max_ps(_mm_mul_ps(_s, _src), _src));
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

        void NeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                NeuralDerivativeRelu<true>(src, size, slope, dst);
            else
                NeuralDerivativeRelu<false>(src, size, slope, dst);
        }

        template <bool align> SIMD_INLINE void UpdateWeights(const float * x, const __m128 & a, const __m128 & b, float * d, float * w)
        {
            __m128 _d = _mm_add_ps(_mm_mul_ps(a, Load<align>(d)), _mm_mul_ps(b, Load<align>(x)));
            Store<align>(d, _d);
            Store<align>(w, _mm_add_ps(Load<align>(w), _d));
        }

        template <bool align> SIMD_INLINE void UpdateWeights(const float * x, size_t offset, const __m128 & a, const __m128 & b, float * d, float * w)
        {
            UpdateWeights<align>(x + offset, a, b, d + offset, w + offset);
        }

        template <bool align> void NeuralUpdateWeights(const float * x, size_t size, const float & a, const float & b, float * d, float * w)
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

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float * delta, const __m128 & norm, const __m128 & alpha, const __m128 & epsilon, float * gradient, float * weight)
        {
            __m128 d = _mm_mul_ps(Load<align>(delta), norm);
            __m128 _gradient = _mm_add_ps(Load<align>(gradient), _mm_mul_ps(d, d));
            Store<align>(gradient, _gradient);
            Store<align>(weight, _mm_sub_ps(Load<align>(weight), _mm_mul_ps(_mm_mul_ps(alpha, d), _mm_rsqrt_ps(_mm_add_ps(_gradient, epsilon)))));
        }

        template <bool align> SIMD_INLINE void AdaptiveGradientUpdate(const float * delta, size_t offset, const __m128 & norm, const __m128 & alpha, const __m128 & epsilon, float * gradient, float * weight)
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

        void NeuralAdaptiveGradientUpdate(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight)
        {
            if (Aligned(delta) && Aligned(gradient) && Aligned(weight))
                NeuralAdaptiveGradientUpdate<true>(delta, size, batch, alpha, epsilon, gradient, weight);
            else
                NeuralAdaptiveGradientUpdate<false>(delta, size, batch, alpha, epsilon, gradient, weight);
        }

        template <size_t size> SIMD_INLINE void LoadWeights(const float * src, __m128 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm_set1_ps(src[i]);
        }

        template <bool align> SIMD_INLINE __m128 Convolution3(const float * src, const __m128 * weights)
        {
            return _mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]),
                _mm_add_ps(_mm_mul_ps(Load<false>(src + 1), weights[1]),
                    _mm_mul_ps(Load<false>(src + 2), weights[2])));
        }

        template <bool align> SIMD_INLINE __m128 Convolution3x3(const float * src, size_t stride, const __m128 * weights)
        {
            return _mm_add_ps(Convolution3<align>(src, weights),
                _mm_add_ps(Convolution3<align>(src + stride, weights + 3),
                Convolution3<align>(src + 2 * stride, weights + 6)));
        }

        template <bool align> void NeuralAddConvolution3x3(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _weights[9];
            LoadWeights<9>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    _dst = _mm_add_ps(_dst, Convolution3x3<align>(src + col, srcStride, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m128 _dst = Load<false>(dst + col);
                    _dst = _mm_add_ps(_dst, _mm_and_ps(tailMask, Convolution3x3<false>(src + col, srcStride, _weights)));
                    Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution3x3(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution3x3<true>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolution3x3<false>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align> SIMD_INLINE __m128 Convolution5(const float * src, const __m128 * weights)
        {
            return _mm_add_ps(_mm_mul_ps(Load<align>(src), weights[0]), _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(Load<false>(src + 1), weights[1]), _mm_mul_ps(Load<false>(src + 2), weights[2])),
                _mm_add_ps(_mm_mul_ps(Load<false>(src + 3), weights[3]), _mm_mul_ps(Load<align>(src + 4), weights[4]))));
        }

        template <bool align> SIMD_INLINE __m128 Convolution5x5(const float * src, size_t stride, const __m128 * weights)
        {
            return _mm_add_ps(Convolution5<align>(src, weights), _mm_add_ps(
                _mm_add_ps(Convolution5<align>(src + stride, weights + 5), Convolution5<align>(src + 2*stride, weights + 10)),
                _mm_add_ps(Convolution5<align>(src + 3*stride, weights + 15), Convolution5<align>(src + 4*stride, weights + 20))));
        }

        template <bool align> void NeuralAddConvolution5x5(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _weights[25];
            LoadWeights<25>(weights, _weights);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    _dst = _mm_add_ps(_dst, Convolution5x5<align>(src + col, srcStride, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m128 _dst = Load<false>(dst + col);
                    _dst = _mm_add_ps(_dst, _mm_and_ps(tailMask, Convolution5x5<false>(src + col, srcStride, _weights)));
                    Store<false>(dst + col, _dst);
                }                
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution5x5(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution5x5<true>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolution5x5<false>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align> void NeuralAddConvolution3x3BackSmall(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t aligned = AlignLo(width, QF);
            size_t partial = AlignLo(width, F);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < 3; ++dy)
                {
                    const float * w = weights + dy * 3;
                    float * d = dst + dy*dstStride;
                    AddMultiplied<align>(src, aligned, partial, width, w[0], d + 0);
                    AddMultiplied<false>(src, aligned, partial, width, w[1], d + 1);
                    AddMultiplied<false>(src, aligned, partial, width, w[2], d + 2);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        namespace
        {
            template<int half> struct Buffer
            {
                Buffer(size_t width)
                {
                    _count = 1 + 2 * half;
                    _size = width*sizeof(float);
                    size_t stride = AlignHi(width + 4 * half, F);
                    size_t full = _count*stride*sizeof(float);
                    _ptr = Allocate(full);
                    memset(_ptr, 0, full);
                    rows[0] = (float*)_ptr;
                    for (size_t i = 1; i < _count; ++i)
                        rows[i] = rows[i - 1] + stride;
                }

                void Update(const float * src)
                {
                    float * tmp = rows[0];
                    if (src == NULL)
                        memset(tmp + 2*half, 0, _size);
                    else
                        memcpy(tmp + 2*half, src, _size);
                    for (size_t i = 0; i < _count - 1; ++i)
                        rows[i] = rows[i + 1];
                    rows[_count - 1] = tmp;
                }

                ~Buffer()
                {
                    Free(_ptr);
                }

                float * rows[1 + 2*half];
            private:
                size_t _count, _size;
                void * _ptr;
            };
        }

        template <size_t size> SIMD_INLINE void LoadWeightsBack(const float * src, __m128 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm_set1_ps(src[size - i - 1]);
        }

        template <bool align, int half> SIMD_INLINE __m128 Convolution3x3Back(const Buffer<half> & buffer, size_t offset, const __m128 * weights)
        {
            return _mm_add_ps(Convolution3<align>(buffer.rows[0] + offset, weights),
                _mm_add_ps(Convolution3<align>(buffer.rows[1] + offset, weights + 3),
                    Convolution3<align>(buffer.rows[2] + offset, weights + 6)));
        }

        template <bool align> void NeuralAddConvolution3x3BackLarge(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            Buffer<1> buffer(width);
            height += 2;
            width += 2;
            size_t alignedWidth = AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _weights[9];
            LoadWeightsBack<9>(weights, _weights);

            for (size_t row = 0; row < height; ++row)
            {
                buffer.Update(row < height - 2 ? src : NULL);
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    _dst = _mm_add_ps(_dst, Convolution3x3Back<true>(buffer, col, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m128 _dst = Load<false>(dst + col);
                    _dst = _mm_add_ps(_dst, _mm_and_ps(tailMask, Convolution3x3Back<false>(buffer, col, _weights)));
                    Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align> void NeuralAddConvolution3x3Back(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (width*height < 1024)
                NeuralAddConvolution3x3BackSmall<align>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolution3x3BackLarge<align>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Back(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution3x3Back<true>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolution3x3Back<false>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align> void NeuralAddConvolution5x5BackSmall(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t aligned = AlignLo(width, QF);
            size_t partial = AlignLo(width, F);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < 5; ++dy)
                {
                    const float * w = weights + dy * 5;
                    float * d = dst + dy*dstStride;
                    AddMultiplied<align>(src, aligned, partial, width, w[0], d + 0);
                    AddMultiplied<false>(src, aligned, partial, width, w[1], d + 1);
                    AddMultiplied<false>(src, aligned, partial, width, w[2], d + 2);
                    AddMultiplied<false>(src, aligned, partial, width, w[3], d + 3);
                    AddMultiplied<align>(src, aligned, partial, width, w[4], d + 4);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align, int half> SIMD_INLINE __m128 Convolution5x5Back(const Buffer<half> & buffer, size_t offset, const __m128 * weights)
        {
            return _mm_add_ps(_mm_add_ps(Convolution5<align>(buffer.rows[0] + offset, weights),
                _mm_add_ps(Convolution5<align>(buffer.rows[1] + offset, weights + 5), 
                    Convolution5<align>(buffer.rows[2] + offset, weights + 10))),
                _mm_add_ps(Convolution5<align>(buffer.rows[3] + offset, weights + 15), 
                    Convolution5<align>(buffer.rows[4] + offset, weights + 20)));
        }

        template <bool align> void NeuralAddConvolution5x5BackLarge(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            Buffer<2> buffer(width);
            height += 4;
            width += 4;
            size_t alignedWidth = AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _weights[25];
            LoadWeightsBack<25>(weights, _weights);

            for (size_t row = 0; row < height; ++row)
            {
                buffer.Update(row < height - 4 ? src : NULL);
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    _dst = _mm_add_ps(_dst, Convolution5x5Back<true>(buffer, col, _weights));
                    Store<align>(dst + col, _dst);
                }
                if (width - alignedWidth)
                {
                    size_t col = width - F;
                    __m128 _dst = Load<false>(dst + col);
                    _dst = _mm_add_ps(_dst, _mm_and_ps(tailMask, Convolution5x5Back<false>(buffer, col, _weights)));
                    Store<false>(dst + col, _dst);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        template <bool align> void NeuralAddConvolution5x5Back(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (width*height < 2048)
                NeuralAddConvolution5x5BackSmall<align>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolution5x5BackLarge<align>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Back(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution5x5Back<true>(src, srcStride, width, height, weights, dst, dstStride);
            else
                NeuralAddConvolution5x5Back<false>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <bool align> SIMD_INLINE void AddMultiplied3(const float * src, const __m128 & dst, __m128 * sums)
        {
            sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
            sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
            sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied3x3(const float * src, size_t stride, const __m128 & dst, __m128 * sums)
        {
            AddMultiplied3<align>(src + stride * 0, dst, sums + 0);
            AddMultiplied3<align>(src + stride * 1, dst, sums + 3);
            AddMultiplied3<align>(src + stride * 2, dst, sums + 6);
        }

        template <bool align> void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _sums[9];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    AddMultiplied3x3<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m128 _dst = _mm_and_ps(tailMask, Load<false>(dst + col));
                    AddMultiplied3x3<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            for (size_t i = 0; i < 9; ++i)
                sums[i] += ExtractSum(_sums[i]);
        }

        void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution3x3Sum<true>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolution3x3Sum<false>(src, srcStride, dst, dstStride, width, height, sums);
        }

        template <bool align> SIMD_INLINE void AddMultiplied5(const float * src, const __m128 & dst, __m128 * sums)
        {
            sums[0] = _mm_add_ps(sums[0], _mm_mul_ps(dst, Load<align>(src + 0)));
            sums[1] = _mm_add_ps(sums[1], _mm_mul_ps(dst, Load<false>(src + 1)));
            sums[2] = _mm_add_ps(sums[2], _mm_mul_ps(dst, Load<false>(src + 2)));
            sums[3] = _mm_add_ps(sums[3], _mm_mul_ps(dst, Load<false>(src + 3)));
            sums[4] = _mm_add_ps(sums[4], _mm_mul_ps(dst, Load<align>(src + 4)));
        }

        template <bool align> SIMD_INLINE void AddMultiplied5x5(const float * src, size_t stride, const __m128 & dst, __m128 * sums)
        {
            AddMultiplied5<align>(src + stride * 0, dst, sums + 0);
            AddMultiplied5<align>(src + stride * 1, dst, sums + 5);
            AddMultiplied5<align>(src + stride * 2, dst, sums + 10);
            AddMultiplied5<align>(src + stride * 3, dst, sums + 15);
            AddMultiplied5<align>(src + stride * 4, dst, sums + 20);
        }

        template <bool align> void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t alignedWidth = Simd::AlignLo(width, F);
            __m128 tailMask = RightNotZero(width - alignedWidth);
            __m128 _sums[25];
            memset(_sums, 0, sizeof(_sums));
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    __m128 _dst = Load<align>(dst + col);
                    AddMultiplied5x5<align>(src + col, srcStride, _dst, _sums);
                }
                if (alignedWidth < width)
                {
                    size_t col = width - F;
                    __m128 _dst = _mm_and_ps(tailMask, Load<false>(dst + col));
                    AddMultiplied5x5<false>(src + col, srcStride, _dst, _sums);
                }
                src += srcStride;
                dst += dstStride;
            }
            for (size_t i = 0; i < 25; ++i)
                sums[i] += ExtractSum(_sums[i]);
        }

        void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralAddConvolution5x5Sum<true>(src, srcStride, dst, dstStride, width, height, sums);
            else
                NeuralAddConvolution5x5Sum<false>(src, srcStride, dst, dstStride, width, height, sums);
        }

        template <bool align> SIMD_INLINE __m128 Max2x2(const float * src, size_t stride)
        {
            __m128 _src0 = _mm_max_ps(Load<align>(src + 0), Load<align>(src + stride + 0));
            __m128 _src1 = _mm_max_ps(Load<align>(src + F), Load<align>(src + stride + F));
            return _mm_max_ps(_mm_shuffle_ps(_src0, _src1, 0x88), _mm_shuffle_ps(_src0, _src1, 0xDD));
        }

        template <bool align> void NeuralMax2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, DF);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                {
                    Store<align>(dst + (col >> 1), Max2x2<align>(src + col, srcStride));
                }
                if (width - alignedWidth)
                {
                    size_t col = width - DF;
                    Store<false>(dst + (col >> 1), Max2x2<false>(src + col, srcStride));
                }
                src += 2*srcStride;
                dst += dstStride;
            }
        }

        void NeuralMax2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride, F) && Aligned(dst) && Aligned(dstStride, F))
                NeuralMax2x2<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralMax2x2<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif// SIMD_SSE_ENABLE
}

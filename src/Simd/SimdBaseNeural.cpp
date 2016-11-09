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
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
		void NeuralConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion)
		{
            const float k = 1.0f / 255.0f;
			for (size_t row = 0; row < height; ++row)
			{
				if (inversion)
				{
					for (size_t col = 0; col < width; ++col)
						dst[col] = (255 - src[col])* k;
				}
				else
				{
					for (size_t col = 0; col < width; ++col)
						dst[col] = src[col] * k;
				}
				src += stride;
				dst += width;
			}
		}

        SIMD_INLINE float ProductSum(const float * a, const float * b, size_t aligned, size_t full)
        {
            size_t i = 0;
            float sums[4] = { 0, 0, 0, 0 };
            for (; i < aligned; i += 4)
            {
                sums[0] += a[i + 0] * b[i + 0];
                sums[1] += a[i + 1] * b[i + 1];
                sums[2] += a[i + 2] * b[i + 2];
                sums[3] += a[i + 3] * b[i + 3];
            }
            for (; i < full; ++i)
                sums[0] += a[i] * b[i];
            return sums[0] + sums[1] + sums[2] + sums[3];
        }

        void NeuralProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            *sum = ProductSum(a, b, Simd::AlignLo(size, 4), size);
        }

        SIMD_INLINE void AddMultiplied(const float * src, size_t aligned, size_t full, float value, float * dst)
        {
            size_t i = 0;
            for (; i < aligned; i += 4)
            {
                dst[i + 0] += src[i + 0] * value;
                dst[i + 1] += src[i + 1] * value;
                dst[i + 2] += src[i + 2] * value;
                dst[i + 3] += src[i + 3] * value;
            }
            for (; i < full; ++i)
                dst[i] += src[i] * value;
        }

        void NeuralAddVectorMultipliedByValue(const float * src, size_t size, const float * value, float * dst)
        {
            AddMultiplied(src, Simd::AlignLo(size, 4), size, *value, dst);
        }

		void NeuralSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			float s = slope[0];
			for (size_t i = 0; i < size; ++i)
				dst[i] = Sigmoid(src[i] * s);
		}

		void NeuralRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			float s = slope[0];
			for (size_t i = 0; i < size; ++i)
				dst[i] = RoughSigmoid(src[i] * s);
		}

        void NeuralRoughSigmoid2(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = RoughSigmoid2(src[i] * s);
        }

        void NeuralDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] *= s*DerivativeSigmoid(src[i]);
        }

        void NeuralTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = Tanh(src[i] * s);
        }

        void NeuralRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = RoughTanh(src[i] * s);
        }

        void NeuralDerivativeTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] *= s*DerivativeTanh(src[i]);
        }

        void NeuralRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            assert(s >= 0.0f && s <= 1.0f);
            if (s == 0)
            {
                for (size_t i = 0; i < size; ++i)
                    dst[i] = Simd::Max(0.0f, src[i]);
            }
            else
            {
                for (size_t i = 0; i < size; ++i)
                    dst[i] = Simd::Max(src[i]*s, src[i]);
            }
        }

        void NeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] *= src[i] > 0 ? 1.0f : s;
        }

        void NeuralUpdateWeights(const float * x, size_t size, const float * a, const float * b, float * d, float * w)
        {
            float _a = a[0], _b = b[0];
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                UpdateWeights(x, i + 0, _a, _b, d, w);
                UpdateWeights(x, i + 1, _a, _b, d, w);
                UpdateWeights(x, i + 2, _a, _b, d, w);
                UpdateWeights(x, i + 3, _a, _b, d, w);
            }
            for (; i < size; ++i)
                UpdateWeights(x, i, _a, _b, d, w);
        }

        void NeuralAdaptiveGradientUpdate(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight)
        {
            float norm = (float)(1.0/batch), _alpha = alpha[0], _epsilon = epsilon[0];
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                AdaptiveGradientUpdate(delta, i + 0, norm, _alpha, _epsilon, gradient, weight);
                AdaptiveGradientUpdate(delta, i + 1, norm, _alpha, _epsilon, gradient, weight);
                AdaptiveGradientUpdate(delta, i + 2, norm, _alpha, _epsilon, gradient, weight);
                AdaptiveGradientUpdate(delta, i + 3, norm, _alpha, _epsilon, gradient, weight);
            }
            for (; i < size; ++i)
                AdaptiveGradientUpdate(delta, i, norm, _alpha, _epsilon, gradient, weight);
        }

        SIMD_INLINE float Convolution3(const float * src, const float * weights)
        {
            return src[0] * weights[0] + src[1] * weights[1] + src[2] * weights[2];
        }

        SIMD_INLINE float Convolution3x3(const float * src, size_t stride, const float * weights)
        {
            return
                Convolution3(src, weights) +
                Convolution3(src + stride, weights + 3) +
                Convolution3(src + 2 * stride, weights + 6);
        }

        void NeuralAddConvolution3x3(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    dst[col] += Convolution3x3(src + col, srcStride, weights);
                src += srcStride;
                dst += dstStride;
            }
        }

        SIMD_INLINE float Convolution5(const float * src, const float * weights)
        {
            return src[0] * weights[0] + src[1] * weights[1] + src[2] * weights[2] + src[3] * weights[3] + src[4] * weights[4];
        }

        SIMD_INLINE float Convolution5x5(const float * src, size_t stride, const float * weights)
        {
            return
                Convolution5(src, weights) +
                Convolution5(src + stride, weights + 5) +
                Convolution5(src + 2 * stride, weights + 10) +
                Convolution5(src + 3 * stride, weights + 15) +
                Convolution5(src + 4 * stride, weights + 20);
        }

        void NeuralAddConvolution5x5(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    dst[col] += Convolution5x5(src + col, srcStride, weights);
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution3x3Back(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t aligned = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < 3; ++dy)
                {
                    const float * w = weights + dy * 3;
                    float * d = dst + dy*dstStride;
                    for (size_t dx = 0; dx < 3; ++dx)
                        AddMultiplied(src, aligned, width, w[dx], d + dx);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution5x5Back(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t aligned = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < 5; ++dy)
                {
                    const float * w = weights + dy * 5;
                    float * d = dst + dy*dstStride;
                    for (size_t dx = 0; dx < 5; ++dx)
                        AddMultiplied(src, aligned, width, w[dx], d + dx);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t aligned = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < 3; ++dy)
                {
                    const float * s = src + dy*srcStride;
                    float * sum = sums + dy * 3;
                    sum[0] += ProductSum(s + 0, dst, aligned, width);
                    sum[1] += ProductSum(s + 1, dst, aligned, width);
                    sum[2] += ProductSum(s + 2, dst, aligned, width);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t aligned = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < 5; ++dy)
                {
                    const float * s = src + dy*srcStride;
                    float * sum = sums + dy * 5;
                    sum[0] += ProductSum(s + 0, dst, aligned, width);
                    sum[1] += ProductSum(s + 1, dst, aligned, width);
                    sum[2] += ProductSum(s + 2, dst, aligned, width);
                    sum[3] += ProductSum(s + 3, dst, aligned, width);
                    sum[4] += ProductSum(s + 4, dst, aligned, width);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        SIMD_INLINE float Max2(const float * src)
        {
            return Simd::Max(src[0], src[1]);
        }

        SIMD_INLINE float Max2x2(const float * src, size_t stride)
        {
            return Simd::Max(Max2(src), Max2(src + stride));
        }

        void NeuralMax2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t col = 0; col < width; col += 2)
                    dst[col>>1] = Max2x2(src + col, srcStride);
                src += 2*srcStride;
                dst += dstStride;
            }
        }
    }
}

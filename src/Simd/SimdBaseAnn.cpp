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
		void AnnConvert(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion)
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

        void AnnProductSum(const float * a, const float * b, size_t size, float * sum)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            float sums[4] = {0, 0, 0, 0};
            size_t i = 0;
            for(; i < alignedSize; i += 4)
            {
                sums[0] += a[i + 0]*b[i + 0];
                sums[1] += a[i + 1]*b[i + 1];
                sums[2] += a[i + 2]*b[i + 2];
                sums[3] += a[i + 3]*b[i + 3];
            }
            for(; i < size; ++i)
                sums[0] += a[i]*b[i];
            *sum = sums[0] + sums[1] + sums[2] + sums[3];
        }

        void AnnAddVectorMultiplyedByValue(const float * src, size_t size, const float * value, float * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            float _value = *value;
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                dst[i + 0] += src[i + 0] * _value;
                dst[i + 1] += src[i + 1] * _value;
                dst[i + 2] += src[i + 2] * _value;
                dst[i + 3] += src[i + 3] * _value;
            }
            for (; i < size; ++i)
                dst[i] += src[i] * _value;
        }

		void AnnSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			float s = slope[0];
			for (size_t i = 0; i < size; ++i)
				dst[i] = Sigmoid(src[i] * s);
		}

		void AnnRoughSigmoid(const float * src, size_t size, const float * slope, float * dst)
		{
			float s = slope[0];
			for (size_t i = 0; i < size; ++i)
				dst[i] = RoughSigmoid(src[i] * s);
		}

        void AnnDerivativeSigmoid(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = s*DerivativeSigmoid(src[i]);
        }

        void AnnTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = Tanh(src[i] * s);
        }

        void AnnRoughTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = RoughTanh(src[i] * s);
        }

        void AnnDerivativeTanh(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = s*DerivativeTanh(src[i]);
        }

        void AnnRelu(const float * src, size_t size, const float * slope, float * dst)
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

        void AnnDerivativeRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = -slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = src[i] > 0 ? 1.0f : s;
        }

        void AnnUpdateWeights(const float * x, size_t size, const float * a, const float * b, float * d, float * w)
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

        void AnnAddConvolution3x3(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
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

        void AnnAddConvolution5x5(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    dst[col] += Convolution5x5(src + col, srcStride, weights);
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

        void AnnMax2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
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

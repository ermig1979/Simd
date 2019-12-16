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
#include "Simd/SimdPow.h"

namespace Simd
{
    namespace Base
    {
        void NeuralConvert(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride, int inversion)
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
                src += srcStride;
                dst += dstStride;
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

        void NeuralAddVector(const float * src, size_t size, float * dst)
        {
            size_t aligned = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < aligned; i += 4)
            {
                dst[i + 0] += src[i + 0];
                dst[i + 1] += src[i + 1];
                dst[i + 2] += src[i + 2];
                dst[i + 3] += src[i + 3];
            }
            for (; i < size; ++i)
                dst[i] += src[i];
        }

        void NeuralAddValue(const float * value, float * dst, size_t size)
        {
            const float val = value[0];
            size_t aligned = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < aligned; i += 4)
            {
                dst[i + 0] += val;
                dst[i + 1] += val;
                dst[i + 2] += val;
                dst[i + 3] += val;
            }
            for (; i < size; ++i)
                dst[i] += val;
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

        void NeuralDerivativeRelu(const float * src, size_t size, const float * slope, float * dst)
        {
            float s = slope[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] *= src[i] > 0 ? 1.0f : s;
        }

        void NeuralPow(const float * src, size_t size, const float * exponent, float * dst)
        {
            float e = exponent[0];
            for (size_t i = 0; i < size; ++i)
                dst[i] = Pow(src[i], e);
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
            float norm = (float)(1.0 / batch), _alpha = alpha[0], _epsilon = epsilon[0];
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

        SIMD_INLINE float Convolution2(const float * src, const float * weights)
        {
            return src[0] * weights[0] + src[1] * weights[1];
        }

        SIMD_INLINE float Convolution2x2Forward(const float * src, size_t stride, const float * weights)
        {
            return
                Convolution2(src, weights) +
                Convolution2(src + stride, weights + 2);
        }

        void NeuralAddConvolution2x2Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    dst[col] += Convolution2x2Forward(src + col, srcStride, weights);
                src += srcStride;
                dst += dstStride;
            }
        }

        SIMD_INLINE float Convolution3(const float * src, const float * weights)
        {
            return src[0] * weights[0] + src[1] * weights[1] + src[2] * weights[2];
        }

        SIMD_INLINE float Convolution3x3Forward(const float * src, size_t stride, const float * weights)
        {
            return
                Convolution3(src, weights) +
                Convolution3(src + stride, weights + 3) +
                Convolution3(src + 2 * stride, weights + 6);
        }

        void NeuralAddConvolution3x3Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    dst[col] += Convolution3x3Forward(src + col, srcStride, weights);
                src += srcStride;
                dst += dstStride;
            }
        }

        SIMD_INLINE float Convolution4(const float * src, const float * weights)
        {
            return src[0] * weights[0] + src[1] * weights[1] + src[2] * weights[2] + src[3] * weights[3];
        }

        SIMD_INLINE float Convolution4x4Forward(const float * src, size_t stride, const float * weights)
        {
            return
                Convolution4(src, weights) +
                Convolution4(src + stride, weights + 4) +
                Convolution4(src + 2 * stride, weights + 8) +
                Convolution4(src + 3 * stride, weights + 12);
        }

        void NeuralAddConvolution4x4Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    dst[col] += Convolution4x4Forward(src + col, srcStride, weights);
                src += srcStride;
                dst += dstStride;
            }
        }

        SIMD_INLINE float Convolution5(const float * src, const float * weights)
        {
            return src[0] * weights[0] + src[1] * weights[1] + src[2] * weights[2] + src[3] * weights[3] + src[4] * weights[4];
        }

        SIMD_INLINE float Convolution5x5Forward(const float * src, size_t stride, const float * weights)
        {
            return
                Convolution5(src, weights) +
                Convolution5(src + stride, weights + 5) +
                Convolution5(src + 2 * stride, weights + 10) +
                Convolution5(src + 3 * stride, weights + 15) +
                Convolution5(src + 4 * stride, weights + 20);
        }

        void NeuralAddConvolution5x5Forward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    dst[col] += Convolution5x5Forward(src + col, srcStride, weights);
                src += srcStride;
                dst += dstStride;
            }
        }

        template <size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionBackward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            size_t aligned = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < coreY; ++dy)
                {
                    const float * w = weights + dy * coreX;
                    float * d = dst + dy*dstStride;
                    for (size_t dx = 0; dx < coreX; ++dx)
                        AddMultiplied(src, aligned, width, w[dx], d + dx);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution2x2Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            NeuralAddConvolutionBackward<2, 2>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution3x3Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            NeuralAddConvolutionBackward<3, 3>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution4x4Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            NeuralAddConvolutionBackward<4, 4>(src, srcStride, width, height, weights, dst, dstStride);
        }

        void NeuralAddConvolution5x5Backward(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride)
        {
            NeuralAddConvolutionBackward<5, 5>(src, srcStride, width, height, weights, dst, dstStride);
        }

        template <size_t coreX, size_t coreY> SIMD_INLINE void NeuralAddConvolutionSum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            size_t aligned = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t dy = 0; dy < coreY; ++dy)
                {
                    const float * s = src + dy*srcStride;
                    float * sum = sums + dy * coreX;
                    for (size_t dx = 0; dx < coreX; ++dx)
                        sum[dx] += ProductSum(s + dx, dst, aligned, width);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralAddConvolution2x2Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            NeuralAddConvolutionSum<2, 2>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution3x3Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            NeuralAddConvolutionSum<3, 3>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution4x4Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            NeuralAddConvolutionSum<4, 4>(src, srcStride, dst, dstStride, width, height, sums);
        }

        void NeuralAddConvolution5x5Sum(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums)
        {
            NeuralAddConvolutionSum<5, 5>(src, srcStride, dst, dstStride, width, height, sums);
        }

        SIMD_INLINE float Max2(const float * src)
        {
            return Simd::Max(src[0], src[1]);
        }

        SIMD_INLINE float Max2x2(const float * src, size_t stride)
        {
            return Simd::Max(Max2(src), Max2(src + stride));
        }

        SIMD_INLINE float Max3(const float * src)
        {
            return Simd::Max(src[0], Simd::Max(src[1], src[2]));
        }

        SIMD_INLINE float Max3x3(const float * src, size_t stride)
        {
            return Simd::Max(Max3(src), Simd::Max(Max3(src + stride), Max3(src + 2 * stride)));
        }

        SIMD_INLINE float Max2x3(const float * src, size_t stride)
        {
            return Simd::Max(Max2(src), Simd::Max(Max2(src + stride), Max2(src + 2 * stride)));
        }

        SIMD_INLINE float Max3x2(const float * src, size_t stride)
        {
            return Simd::Max(Max3(src), Max3(src + stride));
        }

        void NeuralPooling1x1Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            src -= 1;

            dst[0] = Max2x2(src + 1, srcStride);
            for (size_t col = 1; col < width; ++col)
                dst[col] = Max3x2(src + col, srcStride);
            dst[width] = Max2x2(src + width, srcStride);
            dst += dstStride;

            for (size_t row = 1; row < height; ++row)
            {
                dst[0] = Max2x3(src + 1, srcStride);
                for (size_t col = 1; col < width; ++col)
                    dst[col] = Max3x3(src + col, srcStride);
                dst[width] = Max2x3(src + width, srcStride);
                src += srcStride;
                dst += dstStride;
            }

            dst[0] = Max2x2(src + 1, srcStride);
            for (size_t col = 1; col < width; ++col)
                dst[col] = Max3x2(src + col, srcStride);
            dst[width] = Max2x2(src + width, srcStride);
        }

        void NeuralPooling2x2Max2x2(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < widthEven; col += 2)
                    dst[col >> 1] = Max2x2(src + col, srcStride);
                if (width - widthEven)
                    dst[widthEven >> 1] = Simd::Max(src[widthEven], src[widthEven + srcStride]);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < widthEven; col += 2)
                    dst[col >> 1] = Simd::Max(src[col], src[col + 1]);
                if (width - widthEven)
                    dst[widthEven >> 1] = src[widthEven];
            }
        }

        void NeuralPooling2x2Max3x3(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < widthEven; col += 2)
                    dst[col >> 1] = Max3x3(src + col, srcStride);
                if (width - widthEven)
                    dst[widthEven >> 1] = Max2x3(src + widthEven, srcStride);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < widthEven; col += 2)
                    dst[col >> 1] = Max3x2(src + col, srcStride);
                if (width - widthEven)
                    dst[widthEven >> 1] = Max2x2(src + widthEven, srcStride);
            }
        }

        SIMD_INLINE bool NeuralConvolutionForwardValid(ptrdiff_t a, ptrdiff_t b)
        {
            return size_t(a) < size_t(b);
        }

        void NeuralConvolutionForwardConvertN(const float * src, ptrdiff_t srcWidth, ptrdiff_t srcHeight, ptrdiff_t srcDepth, ptrdiff_t kernelX, ptrdiff_t kernelY,
            ptrdiff_t padX, ptrdiff_t padY, ptrdiff_t strideX, ptrdiff_t strideY, ptrdiff_t dilationX, ptrdiff_t dilationY, float * dst)
        {
            const ptrdiff_t dstHeight = (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            const ptrdiff_t dstWidth = (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            const ptrdiff_t channelSize = srcHeight * srcWidth;
            for (ptrdiff_t channel = 0; channel < srcDepth; ++channel, src += channelSize)
            {
                for (ptrdiff_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                {
                    for (ptrdiff_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                    {
                        ptrdiff_t srcRow = kernelRow*dilationY - padY;
                        for (ptrdiff_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            if (!NeuralConvolutionForwardValid(srcRow, srcHeight))
                            {
                                for (ptrdiff_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                    *(dst++) = 0;
                            }
                            else
                            {
                                ptrdiff_t srcCol = kernelCol*dilationX - padX;
                                for (ptrdiff_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                {
                                    if (NeuralConvolutionForwardValid(srcCol, srcWidth))
                                        *(dst++) = src[srcRow*srcWidth + srcCol];
                                    else
                                        *(dst++) = 0;
                                    srcCol += strideX;
                                }
                            }
                            srcRow += strideY;
                        }
                    }
                }
            }
        }

        void NeuralConvolutionForwardConvertT(const float * src, ptrdiff_t srcWidth, ptrdiff_t srcHeight, ptrdiff_t srcDepth, ptrdiff_t kernelX, ptrdiff_t kernelY,
            ptrdiff_t padX, ptrdiff_t padY, ptrdiff_t strideX, ptrdiff_t strideY, ptrdiff_t dilationX, ptrdiff_t dilationY, float * dst)
        {
            const ptrdiff_t dstHeight = (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            const ptrdiff_t dstWidth = (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            for (ptrdiff_t dstRow = 0; dstRow < dstHeight; ++dstRow)
            {
                ptrdiff_t srcRow0 = dstRow*strideY - padY;
                for (ptrdiff_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                {
                    ptrdiff_t srcCol0 = dstCol*strideX - padX;
                    for (ptrdiff_t channel = 0; channel < srcDepth; ++channel)
                    {
                        ptrdiff_t dstChannelOffset = ((dstRow*dstWidth + dstCol)*srcDepth + channel)*kernelY*kernelX;
                        for (ptrdiff_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                        {
                            ptrdiff_t srcRow = srcRow0 + kernelRow*dilationY;
                            for (ptrdiff_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                            {
                                ptrdiff_t srcCol = srcCol0 + kernelCol*dilationX;
                                ptrdiff_t dstOffset = dstChannelOffset + kernelRow*kernelX + kernelCol;
                                if (NeuralConvolutionForwardValid(srcRow, srcHeight) && NeuralConvolutionForwardValid(srcCol, srcWidth))
                                    dst[dstOffset] = src[(channel*srcHeight + srcRow)*srcWidth + srcCol];
                                else
                                    dst[dstOffset] = 0;
                            }
                        }
                    }
                }
            }
        }

        void NeuralConvolutionForwardGemmNN(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t k = 0; k < K; ++k)
                {
                    float va = a[i*K + k];
                    const float * pb = b + k*N;
                    float * pc = c + i*N;
                    for (size_t j = 0; j < N; ++j)
                        pc[j] += va*pb[j];
                }
            }
        }

        void NeuralConvolutionForwardGemmNT(size_t M, size_t N, size_t K, const float * a, const float * b, float * c)
        {
            for (size_t i = 0; i < M; ++i)
            {
                for (size_t j = 0; j < N; ++j)
                {
                    float s = 0;
                    const float * pa = a + i*K;
                    const float * pb = b + j*K;
                    for (size_t k = 0; k < K; ++k)
                        s += pa[k] * pb[k];
                    c[i*N + j] += s;
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
                memset(dst, 0, dstWidth*dstHeight*dstDepth * sizeof(float));

            float * temporal = NULL;
            void * internal = NULL;

            bool transpose = dstWidth*dstHeight <= 1024;// && srcDepth > 128;

            if (kernelX == 1 && kernelY == 1 && !transpose)
                temporal = (float*)src;
            else
            {
                size_t required = dstWidth*dstHeight*srcDepth*kernelX*kernelY * sizeof(float);
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
                    NeuralConvolutionForwardConvertT(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, temporal);
                else
                    NeuralConvolutionForwardConvertN(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, temporal);
            }

            size_t M = dstDepth, N = dstHeight*dstWidth, K = kernelX*kernelY*srcDepth;
            if (transpose)
                NeuralConvolutionForwardGemmNT(M, N, K, weight, temporal, dst);
            else
                NeuralConvolutionForwardGemmNN(M, N, K, weight, temporal, dst);

            if (internal)
                Free(internal);
        }
    }
}

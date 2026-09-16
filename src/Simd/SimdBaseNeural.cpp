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

namespace Simd
{
    namespace Base
    {
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

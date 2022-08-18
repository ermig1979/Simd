/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#ifndef __SimdSse2_h__
#define __SimdSse2_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        void Gemm32fNN(size_t M, size_t N, size_t K, const float* alpha, const float* A, size_t lda, const float* B, size_t ldb, const float* beta, float* C, size_t ldc);

        void SynetAddBias(const float* bias, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format);

        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst);

        void SynetEltwiseLayerForward(float const* const* src, const float* weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float* dst);

        void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst);

        void SynetHswish32f(const float* src, size_t size, const float* shift, const float* scale, float* dst);

        void SynetInnerProductLayerForward(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst);

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format);

        void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst);

        void SynetPreluLayerForward(const float* src, const float* slope, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format);

        void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst);

        void SynetRestrictRange32f(const float* src, size_t size, const float* lower, const float* upper, float* dst);

        void SynetShuffleLayerForward(const float* src0, const float* src1, size_t channels0, size_t channels1, size_t spatial, float* dst0, float* dst1, SimdTensorFormatType format, int type);

        void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst);

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t size, size_t inner, float * dst);

        void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst);

        void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst);

        void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst);

        void SynetUnaryOperation32fLayerForward(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst);

        void WinogradKernel1x3Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

        void WinogradKernel1x3Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

        void WinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        void WinogradKernel1x5Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

        void WinogradKernel1x5Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

        void WinogradKernel1x5Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        void WinogradKernel2x2Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

        void WinogradKernel2x2Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

        void WinogradKernel2x2Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        void WinogradKernel2x2Block4x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

        void WinogradKernel2x2Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

        void WinogradKernel2x2Block4x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        void WinogradKernel3x3Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

        void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

        void WinogradKernel3x3Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        void WinogradKernel3x3Block3x3SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

        void WinogradKernel3x3Block3x3SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

        void WinogradKernel3x3Block3x3SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

        void WinogradKernel3x3Block4x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans);

        void WinogradKernel3x3Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

        void WinogradKernel3x3Block4x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

    }
#endif// SIMD_SSE2_ENABLE
}
#endif//__SimdSse2_h__

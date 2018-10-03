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
#include "Simd/SimdConvolution.h"
#include "Simd/SimdSse1.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        ConvolutionImgToCol::ConvolutionImgToCol(const ConvParam & p)
            : Base::ConvolutionImgToCol(p)
        {
        }

        void ConvolutionImgToCol::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Sse::Gemm32fNN(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _N, &_0, dst + _dstStep * g, _N);
            if (_bias)
                Sse::SynetAddBias(_bias, p.dstC, p.dstH*p.dstW, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionWinograd2x3p::ConvolutionWinograd2x3p(const ConvParam & p)
            : Base::ConvolutionWinograd2x3p(p)
        {
        }

        void ConvolutionWinograd2x3p::SetWeight(const float * weight, const float * bias)
        {
            const ConvParam & p = _param;
            _weight.Resize(_strideW*_count);
            Sse::Winograd2x3pSetFilter(weight, p.srcC*p.dstC, _weight.data);
            _bias = bias;
        }

        void ConvolutionWinograd2x3p::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count;
            Sse::Winograd2x3pSetInput(src, p.srcC, p.srcH, p.srcW, buf, _pad);
            for (size_t i = 0; i < _count; ++i)
                Sse::Gemm32fNN(_M, _N, _K, &_1, _weight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N);
            Sse::Winograd2x3pSetOutput(bufD, dst, p.dstC, p.dstH, p.dstW);
            if (_bias)
                Sse::SynetAddBias(_bias, p.dstC, p.dstH*p.dstW, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionDirect::ConvolutionDirect(const ConvParam & p)
            : Base::ConvolutionDirect(p)
        {
        }

        void ConvolutionDirect::SetBias(const float * bias, float * dst)
        {
            const ConvParam & p = _param;
            if (p.dstW < F)
            {
                Base::ConvolutionDirect::SetBias(bias, dst);
                return;
            }
            size_t size = p.dstH*p.dstW;
            size_t sizeF = AlignLo(size, F);
            for (size_t i = 0; i < _dstC; ++i)
            {
                __m128 value = _mm_set1_ps(bias[i]);
                for (size_t j = 0; j < sizeF; j += F)
                    _mm_storeu_ps(dst + j, value);
                if(sizeF < size)
                    _mm_storeu_ps(dst + size - F, value);
                dst += size;
            }
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, __m128 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm_set1_ps(src[i]);
        }

        SIMD_INLINE __m128 ConvolutionKernel3Stride1(const float * src, const __m128 * weight)
        {
            return _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src), weight[0]),
                _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(src + 1), weight[1]),
                    _mm_mul_ps(_mm_loadu_ps(src + 2), weight[2])));
        }

        template<bool masked> SIMD_INLINE void AddConvolutionKernel3x3Stride1x1(const float * src, size_t srcW, const __m128  * weight, float * dst, const __m128 & mask)
        {
            __m128 convolution = _mm_add_ps(ConvolutionKernel3Stride1(src, weight),
                _mm_add_ps(ConvolutionKernel3Stride1(src + srcW, weight + 3),
                    ConvolutionKernel3Stride1(src + 2 * srcW, weight + 6)));
            _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), Masked<masked>(convolution, mask)));
        }

        void ConvolutionDirect::AddConvolution(const float * src, const float * weight, float * dst)
        {
            const ConvParam & p = _param;
            if (p.dstW >= F && p.IsStride(1) && p.IsStride(3))
                AddConvolutionKernel3x3Stride1x1(src, weight, dst);
            else
                Base::ConvolutionDirect::AddConvolution(src, weight, dst);
        }

        void ConvolutionDirect::AddConvolutionKernel3x3Stride1x1(const float * src, const float * weight, float * dst)
        {
            const ConvParam & p = _param;
            __m128 _weight[9];
            size_t dstWF = Simd::AlignLo(p.dstW, F);
            __m128 tail = RightNotZero(p.dstW - dstWF);
            for (size_t dc = 0; dc < _dstC; ++dc)
            {
                for (size_t sc = 0; sc < _srcC; ++sc)
                {
                    const float * ps = src + sc * _srcW * _srcH;
                    float * pd = dst + dc * p.dstW * p.dstH;
                    LoadWeight<9>(weight, _weight);
                    for (size_t y = 0; y < p.dstH; ++y)
                    {
                        size_t x = 0;
                        for (; x < dstWF; x += F)
                            Sse::AddConvolutionKernel3x3Stride1x1<false>(ps + x, _srcW, _weight, pd + x, tail);
                        if (dstWF < p.dstW)
                            Sse::AddConvolutionKernel3x3Stride1x1<true>(ps + p.dstW - F, _srcW, _weight, pd + p.dstW - F, tail);
                        ps += _srcW;
                        pd += p.dstW;
                    }
                    weight += p.kernelX*p.kernelY;
                }
            }
        }

        //---------------------------------------------------------------------

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
        {
            ConvParam param(srcC, srcH, srcW, dstC, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group);
            if (ConvolutionWinograd2x3p::Preferable(param))
                return new ConvolutionWinograd2x3p(param);
            else if (ConvolutionDirect::Preferable(param))
                return new ConvolutionDirect(param);
            else
                return new ConvolutionImgToCol(param);
        }
    }
#endif//SIMD_SSE_ENABLE
}

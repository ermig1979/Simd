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
#include "Simd/SimdStore.h"
#include "Simd/SimdAvx512f.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        ConvolutionImgToCol::ConvolutionImgToCol(const ConvParam & p)
            : Avx2::ConvolutionImgToCol(p)
        {
        }

        void ConvolutionImgToCol::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Avx512f::Gemm32fNN(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _N, &_0, dst + _dstStep * g, _N);
            if (_bias)
                Avx512f::SynetAddBias(_bias, p.dstC, p.dstH*p.dstW, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionImgToRow::ConvolutionImgToRow(const ConvParam & p)
            : Avx2::ConvolutionImgToRow(p)
        {
        }

        void ConvolutionImgToRow::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Avx512f::Gemm32fNT(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _K, &_0, dst + _dstStep * g, _N);
            if (_bias)
                Avx512f::SynetAddBias(_bias, p.dstC, p.dstH*p.dstW, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionWinograd2x3p::ConvolutionWinograd2x3p(const ConvParam & p)
            : Avx2::ConvolutionWinograd2x3p(p)
        {
        }

        void ConvolutionWinograd2x3p::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count;
            Avx512f::Winograd2x3pSetInput(src, p.srcC, p.srcH, p.srcW, buf, _pad);
            for (size_t i = 0; i < _count; ++i)
                Avx512f::Gemm32fNN(_M, _N, _K, &_1, _weight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N);
            Avx512f::Winograd2x3pSetOutput(bufD, dst, p.dstC, p.dstH, p.dstW);
            if (_bias)
                Avx512f::SynetAddBias(_bias, p.dstC, p.dstH*p.dstW, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionDirect::ConvolutionDirect(const ConvParam & p)
            : Avx2::ConvolutionDirect(p)
        {
        }

        void ConvolutionDirect::SetBias(const float * bias, float * dst)
        {
            const ConvParam & p = _param;
            size_t size = p.dstH*p.dstW;
            size_t sizeF = AlignLo(size, F);
            __mmask16 tail = TailMask16(size - sizeF);
            for (size_t i = 0; i < _dstC; ++i)
            {
                __m512 value = _mm512_set1_ps(bias[i]);
                size_t j = 0;
                for (; j < sizeF; j += F)
                    _mm512_storeu_ps(dst + j, value);
                if (j < size)
                    _mm512_mask_storeu_ps(dst + j, tail, value);
                dst += size;
            }
        }

        void ConvolutionDirect::AddConvolution(const float * src, const float * weight, float * dst)
        {
            const ConvParam & p = _param;
            if (p.dstW >= DF && p.IsKernel(3) && p.IsStride(1))
                AddConvolutionKernel3x3Stride1x1(src, weight, dst);
            else if (p.dstW >= DF && p.IsKernel(3) && p.IsStride(2))
                AddConvolutionKernel3x3Stride2x2(src, weight, dst);
            else
                Avx2::ConvolutionDirect::AddConvolution(src, weight, dst);
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, __m512 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm512_set1_ps(src[i]);
        }

        SIMD_INLINE __m512 ConvolutionKernel3Stride1(const float * src, const __m512 * weight)
        {
            return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0],
                _mm512_fmadd_ps(_mm512_loadu_ps(src + 1), weight[1],
                    _mm512_mul_ps(_mm512_loadu_ps(src + 2), weight[2])));
        }

        template<bool masked> SIMD_INLINE void AddConvolutionKernel3x3Stride1x1(const float * src, size_t srcW, const __m512  * weight, float * dst, __mmask16 tail)
        {
            __m512 convolution = _mm512_add_ps(ConvolutionKernel3Stride1(src, weight),
                _mm512_add_ps(ConvolutionKernel3Stride1(src + srcW, weight + 3),
                    ConvolutionKernel3Stride1(src + 2 * srcW, weight + 6)));
            Store<false, masked>(dst, _mm512_add_ps((Load<false, masked>(dst, tail)), convolution), tail);
        }

        void ConvolutionDirect::AddConvolutionKernel3x3Stride1x1(const float * src, const float * weight, float * dst)
        {
            const ConvParam & p = _param;
            __m512 _weight[9];
            size_t dstWF = Simd::AlignLo(p.dstW, F);
            __mmask16 tail = TailMask16(p.dstW - dstWF);
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
                            Avx512f::AddConvolutionKernel3x3Stride1x1<false>(ps + x, _srcW, _weight, pd + x, tail);
                        if (x < p.dstW)
                            Avx512f::AddConvolutionKernel3x3Stride1x1<true>(ps + x, _srcW, _weight, pd + x, tail);
                        ps += _srcW;
                        pd += p.dstW;
                    }
                    weight += p.kernelX*p.kernelY;
                }
            }
        }

        SIMD_INLINE __m512 ConvolutionKernel3Stride2(const float * src, const __m512 * weight)
        {
            __m512 s00 = _mm512_loadu_ps(src);
            __m512 s10 = _mm512_loadu_ps(src + F);
            __m512 s02 = _mm512_loadu_ps(src + 2);
            __m512 s12 = _mm512_loadu_ps(src + 2 + F);
            return _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0x88), weight[0],
                _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0xDD), weight[1],
                    _mm512_mul_ps(_mm512_shuffle_ps(s02, s12, 0x88), weight[2])));
        }

        template<bool masked> SIMD_INLINE void AddConvolutionKernel3x3Stride2x2(const float * src, size_t srcW, const __m512  * weight, float * dst, __mmask16 tail)
        {
            __m512 convolution = _mm512_add_ps(ConvolutionKernel3Stride2(src, weight),
                _mm512_add_ps(ConvolutionKernel3Stride2(src + srcW, weight + 3),
                    ConvolutionKernel3Stride2(src + 2 * srcW, weight + 6)));
            Store<false, masked>(dst, _mm512_add_ps((Load<false, masked>(dst, tail)), _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, convolution)), tail);
        }

        void ConvolutionDirect::AddConvolutionKernel3x3Stride2x2(const float * src, const float * weight, float * dst)
        {
            const ConvParam & p = _param;
            __m512 _weight[9];
            size_t dstWF = Simd::AlignLo(p.dstW, F);
            __mmask16 tail = TailMask16(p.dstW - dstWF);
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
                            Avx512f::AddConvolutionKernel3x3Stride2x2<false>(ps + 2 * x, _srcW, _weight, pd + x, tail);
                        if (x < p.dstW)
                            Avx512f::AddConvolutionKernel3x3Stride2x2<true>(ps + 2 * x, _srcW, _weight, pd + x, tail);
                        ps += _srcW * 2;
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
            else if (ConvolutionImgToRow::Preferable(param))
                return new ConvolutionImgToRow(param);
            else if (Base::ConvolutionDirect::Preferable(param))
                return new Avx512f::ConvolutionDirect(param);
            else
                return new ConvolutionImgToCol(param);
        }
    }
#endif//SIMD_AVX512F_ENABLE
}

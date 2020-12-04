/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512f.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"

#if defined(SIMD_X86_ENABLE) && defined(_MSC_VER) && _MSC_VER < 1924
#define SIMD_MSVS2017_WIN32_RELEASE_COMPILER_ERROR
#endif

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst)
        {
#ifdef SIMD_MSVS2017_WIN32_RELEASE_COMPILER_ERROR
            Avx::ConvolutionBiasAndActivation(bias, count, size, activation, params, trans, dst);
#else
            size_t aligned = AlignLo(trans ? count : size, F);
            __mmask16 tail = __mmask16(-1) >> (F + aligned - (trans ? count : size));
            if (activation == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    SynetAddBias(bias, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    __m512 _0 = _mm512_set1_ps(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(dst + i);
                                __m512 _bias = _mm512_loadu_ps(bias + i);
                                _mm512_storeu_ps(dst + i, _mm512_max_ps(_0, _mm512_add_ps(_dst, _bias)));
                            }
                            if (i < count)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + i);
                                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + i);
                                _mm512_mask_storeu_ps(dst + i, tail, _mm512_max_ps(_0, _mm512_add_ps(_dst, _bias)));
                            }
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m512 _bias = _mm512_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(dst + j);
                                _mm512_storeu_ps(dst + j, _mm512_max_ps(_0, _mm512_add_ps(_dst, _bias)));
                            }
                            if (j < size)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + j);
                                _mm512_mask_storeu_ps(dst + j, tail, _mm512_max_ps(_0, _mm512_add_ps(_dst, _bias)));
                            }
                            dst += size;
                        }
                    }
                }
                else
                {
                    float slope = 0;
                    SynetRelu32f(dst, size*count, &slope, dst);
                }
            }
            else if (activation == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    __m512 _slope = _mm512_set1_ps(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(dst + i);
                                __m512 _bias = _mm512_loadu_ps(bias + i);
                                _mm512_storeu_ps(dst + i, SynetRelu32f(_mm512_add_ps(_dst, _bias), _slope));
                            }
                            if (i < count)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + i);
                                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + i);
                                _mm512_mask_storeu_ps(dst + i, tail, SynetRelu32f(_mm512_add_ps(_dst, _bias), _slope));
                            }
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m512 _bias = _mm512_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _bias);
                                _mm512_storeu_ps(dst + j, SynetRelu32f(value, _slope));
                            }
                            if (j < size)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _bias);
                                _mm512_mask_storeu_ps(dst + j, tail, SynetRelu32f(value, _slope));
                            }
                            dst += size;
                        }
                    }
                }
                else
                    SynetRelu32f(dst, size*count, &slope, dst);
            }
            else if (activation == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    __m512 _lower = _mm512_set1_ps(lower);
                    __m512 _upper = _mm512_set1_ps(upper);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + i), _mm512_loadu_ps(bias + i));
                                _mm512_storeu_ps(dst + i, _mm512_min_ps(_mm512_max_ps(_lower, value), _upper));
                            }
                            if (i < count)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + i), _mm512_maskz_loadu_ps(tail, bias + i));
                                _mm512_mask_storeu_ps(dst + i, tail, _mm512_min_ps(_mm512_max_ps(_lower, value), _upper));
                            }
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m512 _bias = _mm512_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _bias);
                                _mm512_storeu_ps(dst + j, _mm512_min_ps(_mm512_max_ps(_lower, value), _upper));
                            }
                            if (j < size)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _bias);
                                _mm512_mask_storeu_ps(dst + j, tail, _mm512_min_ps(_mm512_max_ps(_lower, value), _upper));
                            }
                            dst += size;
                        }
                    }
                }
                else
                    SynetRestrictRange32f(dst, size*count, &lower, &upper, dst);
            }
            else if (activation == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    if (trans)
                    {
                        if (count == 1 || count == 2 || count == 4 || count == 8 || count == 16)
                        {
                            __m512 _bias, _slope;
                            if (count == 1)
                            {
                                _bias = _mm512_broadcast_f32x4(_mm_set1_ps(bias[0]));
                                _slope = _mm512_broadcast_f32x4(_mm_set1_ps(params[0]));
                            }
                            else if (count == 2)
                            {
                                _bias = _mm512_broadcast_f32x4(_mm_setr_ps(bias[0], bias[1], bias[0], bias[1]));
                                _slope = _mm512_broadcast_f32x4(_mm_setr_ps(params[0], params[1], params[0], params[1]));
                            }
                            else if (count == 4)
                            {
                                _bias = _mm512_broadcast_f32x4(_mm_loadu_ps(bias));
                                _slope = _mm512_broadcast_f32x4(_mm_loadu_ps(params));
                            }
                            else if (count == 8)
                            {
                                _bias = _mm512_setr_ps(bias[0], bias[1], bias[2], bias[3], bias[4], bias[5], bias[6], bias[7], 
                                    bias[0], bias[1], bias[2], bias[3], bias[4], bias[5], bias[6], bias[7]);
                                _slope = _mm512_setr_ps(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                                    params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]);
                            }
                            else if (count == 16)
                            {
                                _bias = _mm512_loadu_ps(bias);
                                _slope = _mm512_loadu_ps(params);
                            }
                            else
                                assert(0);
                            size_t n = size * count, nF = AlignLo(n, F), i = 0;
                            for (; i < nF; i += F)
                            {
                                __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + i), _bias);
                                _mm512_storeu_ps(dst + i, SynetRelu32f(value, _slope));
                            }
                            if (i < n)
                            {
                                __mmask16 tail = TailMask16(n - nF);
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + i), _bias);
                                _mm512_mask_storeu_ps(dst + i, tail, SynetRelu32f(value, _slope));
                            }
                        }
                        else
                        {
                            for (size_t j = 0; j < size; ++j)
                            {
                                size_t i = 0;
                                for (; i < aligned; i += F)
                                {
                                    __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + i), _mm512_loadu_ps(bias + i));
                                    _mm512_storeu_ps(dst + i, SynetRelu32f(value, _mm512_loadu_ps(params + i)));
                                }
                                if (i < count)
                                {
                                    __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + i), _mm512_maskz_loadu_ps(tail, bias + i));
                                    _mm512_mask_storeu_ps(dst + i, tail, SynetRelu32f(value, _mm512_maskz_loadu_ps(tail, params + i)));
                                }
                                dst += count;
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m512 _bias = _mm512_set1_ps(bias[i]);
                            __m512 _slope = _mm512_set1_ps(params[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _bias);
                                _mm512_storeu_ps(dst + j, SynetRelu32f(value, _slope));
                            }
                            if (j < size)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _bias);
                                _mm512_mask_storeu_ps(dst + j, tail, SynetRelu32f(value, _slope));
                            }
                            dst += size;
                        }
                    }
                }
                else
                    Avx512f::SynetPreluLayerForward(dst, params, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationElu)
            {
                float alpha = params[0];
                if (bias)
                {
                    __m512 _alpha = _mm512_set1_ps(alpha);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(dst + i);
                                __m512 _bias = _mm512_loadu_ps(bias + i);
                                _mm512_storeu_ps(dst + i, Avx512f::Elu(_mm512_add_ps(_dst, _bias), _alpha));
                            }
                            if (i < count)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + i);
                                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + i);
                                _mm512_mask_storeu_ps(dst + i, tail, Avx512f::Elu(_mm512_add_ps(_dst, _bias), _alpha));
                            }
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m512 _bias = _mm512_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _bias);
                                _mm512_storeu_ps(dst + j, Avx512f::Elu(value, _alpha));
                            }
                            if (j < size)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _bias);
                                _mm512_mask_storeu_ps(dst + j, tail, Avx512f::Elu(value, _alpha));
                            }
                            dst += size;
                        }
                    }
                }
                else
                    SynetElu32f(dst, size*count, &alpha, dst);
            }
            else if (activation == ::SimdConvolutionActivationHswish)
            {
                float shift = params[0];
                float scale = params[1];
                if (bias)
                {
                    __m512 _shift = _mm512_set1_ps(shift);
                    __m512 _scale = _mm512_set1_ps(scale);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(dst + i);
                                __m512 _bias = _mm512_loadu_ps(bias + i);
                                _mm512_storeu_ps(dst + i, Avx512f::SynetHswish32f(_mm512_add_ps(_dst, _bias), _shift, _scale));
                            }
                            if (i < count)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + i);
                                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + i);
                                _mm512_mask_storeu_ps(dst + i, tail, Avx512f::SynetHswish32f(_mm512_add_ps(_dst, _bias), _shift, _scale));
                            }
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m512 _bias = _mm512_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _bias);
                                _mm512_storeu_ps(dst + j, Avx512f::SynetHswish32f(value, _shift, _scale));
                            }
                            if (j < size)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _bias);
                                _mm512_mask_storeu_ps(dst + j, tail, Avx512f::SynetHswish32f(value, _shift, _scale));
                            }
                            dst += size;
                        }
                    }
                }
                else
                    SynetHswish32f(dst, size*count, &shift, &scale, dst);
            }
            else if (activation == ::SimdConvolutionActivationMish)
            {
                float threshold = params[0];
                if (bias)
                {
                    __m512 _threshold = _mm512_set1_ps(threshold);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(dst + i);
                                __m512 _bias = _mm512_loadu_ps(bias + i);
                                _mm512_storeu_ps(dst + i, Avx512f::Mish(_mm512_add_ps(_dst, _bias), _threshold));
                            }
                            if (i < count)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + i);
                                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + i);
                                _mm512_mask_storeu_ps(dst + i, tail, Avx512f::Mish(_mm512_add_ps(_dst, _bias), _threshold));
                            }
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m512 _bias = _mm512_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _bias);
                                _mm512_storeu_ps(dst + j, Avx512f::Mish(value, _threshold));
                            }
                            if (j < size)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _bias);
                                _mm512_mask_storeu_ps(dst + j, tail, Avx512f::Mish(value, _threshold));
                            }
                            dst += size;
                        }
                    }
                }
                else
                    SynetMish32f(dst, size * count, &threshold, dst);
            }
            else
                assert(0);
#endif
        }

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam32f & p)
            : Avx2::SynetConvolution32fGemmNN(p)
        {
            _index.Resize(F);
            for (size_t i = 0; i < F; ++i)
                _index[i] = int(i * p.strideX);
            _nose.Resize(p.kernelX);
            _tail.Resize(p.kernelX);
            ptrdiff_t aligned = AlignHi(p.dstW, F) - F;
            for (size_t kx = 0; kx < p.kernelX; ++kx)
            {
                _nose[kx] = 0;
                _tail[kx] = 0;
                ptrdiff_t sx = kx * p.dilationX - p.padX;
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    if (sx >= 0 && sx < ptrdiff_t(p.srcW) && dx < F)
                        _nose[kx] |= 1 << dx;
                    if (sx < ptrdiff_t(p.srcW) && ptrdiff_t(dx) >= aligned)
                        _tail[kx] |= 1 << (dx - aligned);
                    sx += p.strideX;
                }
            }
            if (p.dstC == 8)
                return;
            _gemm.Init(InitGemmFuncs(Avx512f::Gemm32fNN, "Avx512f", p.gemm, "Ext"));
            if (_param.trans && _param.group == 1)
            {
                if (GemmRuntime())
                {
                    _gemmCb.Init(InitGemmCbFuncs(Avx512f::Gemm32fNNcbBufferSize, Avx512f::Gemm32fNNcbReorderB, Avx512f::Gemm32fNNcbRun, "Avx512f", GemmKernelF2, GemmKernelF3));
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M*_merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Avx512f::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Avx512f::Gemm32fNNcbRun;
                _nhwcReorderB = Avx512f::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = _N > Avx::F ? Avx512f::ConvolutionBiasAndActivation : Avx::ConvolutionBiasAndActivation;
        }

        void SynetConvolution32fGemmNN::ImgToCol(const float * src, float * dst)
        {
            const ConvParam32f & p = _param;
            size_t srcSize = p.srcW * p.srcH;
            if (p.dilationX == 1 && p.dilationY == 1 && p.strideX == 2 && p.strideY == 2 && p.padX == 0 && p.padY == 0 && p.padW == 0 && p.padH == 0 && p.kernelX == 1 && p.kernelY == 1)
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const float * psrc = src + 2 * dy*p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.dilationX*p.dilationY*p.strideX*p.strideY != 1)
            {
                __m512 _0 = _mm512_setzero_ps();
                __m512i index = _mm512_loadu_si512(_index.data);
                size_t aligned = AlignHi(p.dstW, F) - F;
                __mmask16 storeTail = TailMask16(p.dstW - aligned);
                __mmask16 storeNose = aligned ? __mmask16(-1) : storeTail;
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            __mmask16 nose = _nose[kx];
                            __mmask16 tail = _tail[kx];
                            size_t sx0 = kx * p.dilationX - p.padX;
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t dx = 0, sx = sx0 + sy * p.srcW;
                                    _mm512_mask_storeu_ps(dst + dx, storeNose, _mm512_mask_i32gather_ps(_0, nose, index, (src + sx), 4));
                                    dx += F, sx += p.strideX*F;
                                    //if (p.strideX == 3)
                                    //{
                                    //    for (; dx < aligned; dx += F, sx += p.strideX*F)
                                    //        _mm512_storeu_ps(dst + dx, Avx512f::Gather<3>(src + sx));
                                    //}
                                    //else
                                    //{
                                        for (; dx < aligned; dx += F, sx += p.strideX*F)
                                            _mm512_storeu_ps(dst + dx, _mm512_i32gather_ps(index, (src + sx), 4));
                                    //}
                                    if (aligned)
                                        _mm512_mask_storeu_ps(dst + dx, storeTail, _mm512_mask_i32gather_ps(_0, tail, index, (src + sx), 4));
                                }
                                else
                                {
                                    memset(dst, 0, p.dstW * sizeof(float));
                                }
                                dst += p.dstW;
                                sy += p.strideY;
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
                Base::SynetConvolution32fGemmNN::ImgToCol(src, dst);
            }
        }

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam32f & p)
            : Avx2::SynetConvolution32fGemmNT(p)
        {
            _gemm.Init(InitGemmFuncs(Avx512f::Gemm32fNT, "Avx512f"));
            _biasAndActivation = Avx512f::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam32f & p)
            : Avx2::SynetConvolution32fWinograd(p)
        {
            if (p.dstC == 8)
                return;
            if (p.kernelY == 1 && p.kernelX == 3)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Avx512f::WinogradKernel1x3Block1x4SetFilter;
                    _setInput = Avx512f::WinogradKernel1x3Block1x4SetInput;
                    _setOutput = Avx512f::WinogradKernel1x3Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 1 && p.kernelX == 5)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Avx512f::WinogradKernel1x5Block1x4SetFilter;
                    _setInput = Avx512f::WinogradKernel1x5Block1x4SetInput;
                    _setOutput = Avx512f::WinogradKernel1x5Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 2 && p.kernelX == 2)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    SetBlock(4, 4);
                    _setFilter = Avx512f::WinogradKernel2x2Block4x4SetFilter;
                    _setInput = Avx512f::WinogradKernel2x2Block4x4SetInput;
                    _setOutput = Avx512f::WinogradKernel2x2Block4x4SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    SetBlock(2, 2);
                    _setFilter = Avx512f::WinogradKernel2x2Block2x2SetFilter;
                    _setInput = Avx512f::WinogradKernel2x2Block2x2SetInput;
                    _setOutput = Avx512f::WinogradKernel2x2Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else if (p.kernelY == 3 && p.kernelX == 3)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    _setFilter = Avx512f::WinogradKernel3x3Block4x4SetFilter;
                    _setInput = Avx512f::WinogradKernel3x3Block4x4SetInput;
                    _setOutput = Avx512f::WinogradKernel3x3Block4x4SetOutput;
                }
                else if (_blockY == 3 && _blockX == 3)
                {
                    _setFilter = Avx512f::WinogradKernel3x3Block3x3SetFilter;
                    _setInput = Avx512f::WinogradKernel3x3Block3x3SetInput;
                    _setOutput = Avx512f::WinogradKernel3x3Block3x3SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    _setFilter = Avx512f::WinogradKernel3x3Block2x2SetFilter;
                    _setInput = Avx512f::WinogradKernel3x3Block2x2SetInput;
                    _setOutput = Avx512f::WinogradKernel3x3Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else
                assert(0);
            _gemm.Init(InitGemmFuncs(Avx512f::Gemm32fNN, "Avx512f", p.gemm, "Ext"));
            if (_param.trans)
            {
                if (NHWC_GEMM_RUNTIME)
                {
                    _gemmCb.Init(InitGemmCbFuncs(Avx512f::Gemm32fNNcbBufferSize, Avx512f::Gemm32fNNcbReorderB, Avx512f::Gemm32fNNcbRun, "Avx512f", GemmKernelF2, GemmKernelF3));
                    _nhwcStrideW = _gemmCb.At(0).BufferSize(_M*_merge, _N, _K);
                }
                else
                    _nhwcStrideW = Avx512f::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                _nhwcWeight.Resize(_nhwcStrideW*_count);
                _nhwcRun = Avx512f::Gemm32fNNcbRun;
                _nhwcReorderB = Avx512f::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Avx512f::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fDirectNchw::SynetConvolution32fDirectNchw(const ConvParam32f & p)
            : Avx2::SynetConvolution32fDirectNchw(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, __m512 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm512_set1_ps(src[i]);
        }

        template<int kernel, int stride> struct Kernel
        {
            static __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight);
        };

        template<> struct Kernel<1, 1>
        {
            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_mul_ps(_mm512_loadu_ps(src), weight[0]);
            }
        };

        template<> struct Kernel<1, 2>
        {
            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                __m512 s0 = _mm512_loadu_ps(src + 0);
                __m512 s1 = _mm512_loadu_ps(src + F);
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_mul_ps(_mm512_shuffle_ps(s0, s1, 0x88), weight[0]));
            }
        };

        template<> struct Kernel<2, 1>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0],
                    _mm512_mul_ps(_mm512_loadu_ps(src + 1), weight[1]));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2));
            }
        };

        template<> struct Kernel<2, 2>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 s0 = _mm512_loadu_ps(src + 0);
                __m512 s1 = _mm512_loadu_ps(src + F);
                return _mm512_fmadd_ps(_mm512_shuffle_ps(s0, s1, 0x88), weight[0],
                    _mm512_mul_ps(_mm512_shuffle_ps(s0, s1, 0xDD), weight[1]));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2)));
            }
        };

        template<> struct Kernel<3, 1>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0],
                    _mm512_fmadd_ps(_mm512_loadu_ps(src + 1), weight[1],
                        _mm512_mul_ps(_mm512_loadu_ps(src + 2), weight[2])));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight),
                    _mm512_add_ps(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 2>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 s00 = _mm512_loadu_ps(src);
                __m512 s10 = _mm512_loadu_ps(src + F);
                __m512 s02 = _mm512_loadu_ps(src + 2);
                __m512 s12 = _mm512_loadu_ps(src + 2 + F);
                return _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0x88), weight[0],
                    _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0xDD), weight[1],
                        _mm512_mul_ps(_mm512_shuffle_ps(s02, s12, 0x88), weight[2])));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_add_ps(RowConv(src, weight),
                    _mm512_add_ps(RowConv(src + step, weight + 3), RowConv(src + 2 * step, weight + 6))));
            }
        };

        const __m512i K32_IDX_3_0A = SIMD_MM512_SETR_EPI32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
        const __m512i K32_IDX_3_0B = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 7, 10, 13);
        const __m512i K32_IDX_3_1A = SIMD_MM512_SETR_EPI32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46);
        const __m512i K32_IDX_3_1B = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 8, 11, 14);
        const __m512i K32_IDX_3_2A = SIMD_MM512_SETR_EPI32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47);
        const __m512i K32_IDX_3_2B = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 9, 12, 15);

        template<> struct Kernel<3, 3>
        {

            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 src0 = _mm512_loadu_ps(src + 0 * F);
                __m512 src1 = _mm512_loadu_ps(src + 1 * F);
                __m512 src2 = _mm512_loadu_ps(src + 2 * F);
                __m512 s0 = _mm512_mask_permutexvar_ps(_mm512_maskz_permutex2var_ps(0xFFFF, src0, K32_IDX_3_0A, src1), 0xF800, K32_IDX_3_0B, src2);
                __m512 s1 = _mm512_mask_permutexvar_ps(_mm512_maskz_permutex2var_ps(0xFFFF, src0, K32_IDX_3_1A, src1), 0xF800, K32_IDX_3_1B, src2);
                __m512 s2 = _mm512_mask_permutexvar_ps(_mm512_maskz_permutex2var_ps(0xFFFF, src0, K32_IDX_3_2A, src1), 0xFC00, K32_IDX_3_2B, src2);
                return _mm512_fmadd_ps(s0, weight[0], _mm512_fmadd_ps(s1, weight[1], _mm512_mul_ps(s2, weight[2])));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight), _mm512_add_ps(RowConv(src + step, weight + 3), RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<4, 1>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0], _mm512_fmadd_ps(_mm512_loadu_ps(src + 1), weight[1],
                        _mm512_fmadd_ps(_mm512_loadu_ps(src + 2), weight[2], _mm512_mul_ps(_mm512_loadu_ps(src + 3), weight[3]))));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight), _mm512_add_ps(RowConv(src + step, weight + 4),
                    _mm512_add_ps(RowConv(src + 2 * step, weight + 8), RowConv(src + 3 * step, weight + 12))));
            }
        };

        template<> struct Kernel<4, 2>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 s00 = _mm512_loadu_ps(src);
                __m512 s10 = _mm512_loadu_ps(src + F);
                __m512 s02 = _mm512_loadu_ps(src + 2);
                __m512 s12 = _mm512_loadu_ps(src + 2 + F);
                return _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0x88), weight[0], _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0xDD), weight[1],
                    _mm512_fmadd_ps(_mm512_shuffle_ps(s02, s12, 0x88), weight[2], _mm512_mul_ps(_mm512_shuffle_ps(s02, s12, 0xDD), weight[3]))));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_add_ps(RowConv(src, weight),
                    _mm512_add_ps(RowConv(src + step, weight + 4), _mm512_add_ps(RowConv(src + 2 * step, weight + 8), RowConv(src + 3 * step, weight + 12)))));
            }
        };

        template<> struct Kernel<5, 1>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0], _mm512_fmadd_ps(_mm512_loadu_ps(src + 1), weight[1],
                    _mm512_fmadd_ps(_mm512_loadu_ps(src + 2), weight[2], _mm512_fmadd_ps(_mm512_loadu_ps(src + 3), weight[3],
                        _mm512_mul_ps(_mm512_loadu_ps(src + 4), weight[4])))));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight), _mm512_add_ps(RowConv(src + step, weight + 5),
                    _mm512_add_ps(RowConv(src + 2 * step, weight + 10), _mm512_add_ps(RowConv(src + 3 * step, weight + 15), 
                        RowConv(src + 4 * step, weight + 20)))));
            }
        };

        template<> struct Kernel<5, 2>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 s00 = _mm512_loadu_ps(src);
                __m512 s10 = _mm512_loadu_ps(src + F);
                __m512 s02 = _mm512_loadu_ps(src + 2);
                __m512 s12 = _mm512_loadu_ps(src + 2 + F);
                __m512 s04 = _mm512_loadu_ps(src + 4);
                __m512 s14 = _mm512_loadu_ps(src + 4 + F);
                return _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0x88), weight[0], _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0xDD), weight[1],
                    _mm512_fmadd_ps(_mm512_shuffle_ps(s02, s12, 0x88), weight[2], _mm512_fmadd_ps(_mm512_shuffle_ps(s02, s12, 0xDD), weight[3],
                        _mm512_mul_ps(_mm512_shuffle_ps(s04, s14, 0x88), weight[4])))));
            }

            static SIMD_INLINE __m512 SynetConvolution32f(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_add_ps(RowConv(src, weight), _mm512_add_ps(RowConv(src + step, weight + 5), 
                    _mm512_add_ps(RowConv(src + 2 * step, weight + 10), _mm512_add_ps(RowConv(src + 3 * step, weight + 15), RowConv(src + 4 * step, weight + 20))))));
            }
        };

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m512 Activate(__m512 value, const __m512 * params);

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationIdentity>(__m512 value, const __m512 * params)
        {
            return value;
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRelu>(__m512 value, const __m512 * params)
        {
            return _mm512_max_ps(_mm512_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationLeakyRelu>(__m512 value, const __m512 * params)
        {
            return _mm512_add_ps(_mm512_max_ps(_mm512_setzero_ps(), value), _mm512_mul_ps(params[0], _mm512_min_ps(_mm512_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRestrictRange>(__m512 value, const __m512 * params)
        {
            return _mm512_min_ps(_mm512_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationPrelu>(__m512 value, const __m512 * params)
        {
            return _mm512_add_ps(_mm512_max_ps(_mm512_setzero_ps(), value), _mm512_mul_ps(params[0], _mm512_min_ps(_mm512_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationElu>(__m512 value, const __m512 * params)
        {
            return Avx512f::Elu(value, params[0]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHswish>(__m512 value, const __m512 * params)
        {
            return Avx512f::SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationMish>(__m512 value, const __m512* params)
        {
            return Avx512f::Mish(value, params[0]);
        }

        template<int kernel, int stride, ::SimdConvolutionActivationType type> 
        void ConvolutionBiasActivation(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, 
            const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            __m512 _weight[kernel*kernel];
            __m512 _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange || type == ::SimdConvolutionActivationHswish)
                _params[1] = _mm512_set1_ps(params[1]);
            size_t dstWF = Simd::AlignLo(dstW, F);
            __mmask16 tail = TailMask16(dstW - dstWF);
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_set1_ps(params[dc]);
                if (srcC == 1)
                {
                    const float * ps = src;
                    float * pd = dst;
                    LoadWeight<kernel*kernel>(weight, _weight);
                    __m512 _bias = bias ? _mm512_set1_ps(bias[dc]) : _mm512_setzero_ps();
                    for (size_t y = 0; y < dstH; ++y)
                    {
                        size_t x = 0;
                        for (; x < dstWF; x += F)
                        {
                            __m512 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                            _mm512_storeu_ps(pd + x, Activate<type>(_mm512_add_ps(_bias, conv), _params));
                        }
                        if (x < dstW)
                        {
                            __m512 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                            _mm512_mask_storeu_ps(pd + x, tail, Activate<type>(_mm512_add_ps(_bias, conv), _params));
                        }
                        ps += srcW * stride;
                        pd += dstW;
                    }
                    weight += kernel * kernel;
                }
                else
                {
                    size_t sc = 0;
                    for (; sc < 1; ++sc)
                    {
                        const float * ps = src;
                        float * pd = dst;
                        LoadWeight<kernel*kernel>(weight, _weight);
                        __m512 _bias = bias ? _mm512_set1_ps(bias[dc]) : _mm512_setzero_ps();
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            size_t x = 0;
                            for (; x < dstWF; x += F)
                            {
                                __m512 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm512_storeu_ps(pd + x, _mm512_add_ps(_bias, conv));
                            }
                            if (x < dstW)
                            {
                                __m512 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm512_mask_storeu_ps(pd + x, tail, _mm512_add_ps(_bias, conv));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                    for (; sc < srcC - 1; ++sc)
                    {
                        const float * ps = src + sc * srcW * srcH;
                        float * pd = dst;
                        LoadWeight<kernel*kernel>(weight, _weight);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            size_t x = 0;
                            for (; x < dstWF; x += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(pd + x);
                                __m512 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm512_storeu_ps(pd + x, _mm512_add_ps(_dst, conv));
                            }
                            if (x < dstW)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, pd + x);
                                __m512 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm512_mask_storeu_ps(pd + x, tail, _mm512_add_ps(_dst, conv));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                    for (; sc < srcC; ++sc)
                    {
                        const float * ps = src + sc * srcW * srcH;
                        float * pd = dst;
                        LoadWeight<kernel*kernel>(weight, _weight);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            size_t x = 0;
                            for (; x < dstWF; x += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(pd + x);
                                __m512 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm512_storeu_ps(pd + x, Activate<type>(_mm512_add_ps(_dst, conv), _params));
                            }
                            if (x < dstW)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, pd + x);
                                __m512 conv = Kernel<kernel, stride>::SynetConvolution32f(ps + x * stride, srcW, _weight);
                                _mm512_mask_storeu_ps(pd + x, tail, Activate<type>(_mm512_add_ps(_dst, conv), _params));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                }
                dst += dstH * dstW;
            }
        }

         bool SynetConvolution32fDirectNchw::Preferable(const ConvParam32f & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2) || p.IsStride(3)))
                return false;
            double k = double(p.srcC) / p.group * p.strideX * p.strideX * p.strideY / p.kernelX / p.kernelY;
            return k < 2.0 && ((p.IsStride(1) && p.IsKernel(1)) || p.IsKernel(2) || p.IsKernel(3)
#if SIMD_ZMM_COUNT == 32 || 1
                || ((p.IsKernel(4) || p.IsKernel(5)) && p.dstW > F)
#endif
                ) && p.trans == 0;
        }

        template <int kernel, int stride> SynetConvolution32fDirectNchw::ConvolutionBiasActivationPtr SetConvolutionBiasActivation(::SimdConvolutionActivationType type)
        {
            switch (type)
            {
            case ::SimdConvolutionActivationIdentity: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationIdentity>;
            case ::SimdConvolutionActivationRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRelu>;
            case ::SimdConvolutionActivationLeakyRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationLeakyRelu>;
            case ::SimdConvolutionActivationRestrictRange: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRestrictRange>;
            case ::SimdConvolutionActivationPrelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationPrelu>;
            case ::SimdConvolutionActivationElu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationElu>;
            case ::SimdConvolutionActivationHswish: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationHswish>;
            case ::SimdConvolutionActivationMish: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationMish>;
            default:
                assert(0);
                return NULL;
            }
        }

        SynetConvolution32fDirectNchw::ConvolutionBiasActivationPtr SynetConvolution32fDirectNchw::SetConvolutionBiasActivation()
        {
            const ConvParam32f & p = _param;
            if (p.dstW <= HF && p.kernelX <= 3)
                return Avx2::SynetConvolution32fDirectNchw::SetConvolutionBiasActivation();
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Avx512f::SetConvolutionBiasActivation<1, 1>(p.activation);
                if (p.kernelX == 2)
                    return Avx512f::SetConvolutionBiasActivation<2, 1>(p.activation);
                if (p.kernelX == 3)
                    return Avx512f::SetConvolutionBiasActivation<3, 1>(p.activation);
                if (p.kernelX == 4)
                    return Avx512f::SetConvolutionBiasActivation<4, 1>(p.activation);
                if (p.kernelX == 5)
                    return Avx512f::SetConvolutionBiasActivation<5, 1>(p.activation);
                break;
            case 2:
                if (p.kernelX == 2)
                    return Avx512f::SetConvolutionBiasActivation<2, 2>(p.activation);
                if (p.kernelX == 3)
                    return Avx512f::SetConvolutionBiasActivation<3, 2>(p.activation);
                if (p.kernelX == 4)
                    return Avx512f::SetConvolutionBiasActivation<4, 2>(p.activation);
                if (p.kernelX == 5)
                    return Avx512f::SetConvolutionBiasActivation<5, 2>(p.activation);
                break;
            case 3:
                if (p.kernelX == 3)
                    return Avx512f::SetConvolutionBiasActivation<3, 3>(p.activation);
                break;
            }
            return Avx2::SynetConvolution32fDirectNchw::SetConvolutionBiasActivation();
        }

        //---------------------------------------------------------------------

        SynetConvolution32fDirectNhwc::SynetConvolution32fDirectNhwc(const ConvParam32f & p)
            : Avx2::SynetConvolution32fDirectNhwc(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        SIMD_INLINE void KernelHwcDefaultEdge(const float * src, const ConvParam32f & p, size_t kH, size_t kW, const float * weight, __m512 & sum, __mmask16 tail = -1)
        {
            size_t size = kW * p.srcC, rest = (p.kernelX - kW)*p.srcC*p.dstC, dstC = p.dstC, stride = p.srcW * p.srcC;
            for (size_t ky = 0; ky < kH; ++ky)
            {
                for (size_t i = 0; i < size; ++i, weight += dstC)
                    sum = _mm512_fmadd_ps(_mm512_set1_ps(src[i]), _mm512_maskz_loadu_ps(tail, weight), sum);
                weight += rest;
                src += stride;
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultEdge(const float * src, const ConvParam32f & p, size_t kH, size_t kW, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstC = p.dstC;
            size_t dstCF = AlignLo(dstC, F);

            size_t dc = 0;
            for (; dc < dstCF; dc += F)
            {
                __m512 conv = bias ? _mm512_loadu_ps(bias + dc) : _mm512_setzero_ps();
                KernelHwcDefaultEdge(src, p, kH, kW, weight + dc, conv);
                _mm512_storeu_ps(dst + dc, Activate<type>(conv, params, dc));
            }
            if (dc < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF); 
                __m512 conv = bias ? _mm512_maskz_loadu_ps(tail, bias + dc) : _mm512_setzero_ps();
                KernelHwcDefaultEdge(src, p, kH, kW, weight + dc, conv, tail);
                _mm512_mask_storeu_ps(dst + dc, tail, Activate<type>(conv, params, dc, tail));
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody2x2(const float * src, const ConvParam32f & p, const float * weight, __m512 sums[2][2])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            __m512 w0, w1, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm512_loadu_ps(weight + 0 * F);
                    w1 = _mm512_loadu_ps(weight + 1 * F);
                    s0 = _mm512_set1_ps(src0[offset]);
                    sums[0][0] = _mm512_fmadd_ps(s0, w0, sums[0][0]);
                    sums[0][1] = _mm512_fmadd_ps(s0, w1, sums[0][1]);
                    s0 = _mm512_set1_ps(src1[offset]);
                    sums[1][0] = _mm512_fmadd_ps(s0, w0, sums[1][0]);
                    sums[1][1] = _mm512_fmadd_ps(s0, w1, sums[1][1]);
                    weight += dstC;
                }
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody2x1(const float * src, const ConvParam32f & p, const float * weight, __m512 sums[2][1], __mmask16 tail = -1)
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            __m512 w0, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm512_maskz_loadu_ps(tail, weight + 0 * F);
                    s0 = _mm512_set1_ps(src0[offset]);
                    sums[0][0] = _mm512_fmadd_ps(s0, w0, sums[0][0]);
                    s0 = _mm512_set1_ps(src1[offset]);
                    sums[1][0] = _mm512_fmadd_ps(s0, w0, sums[1][0]);
                    weight += dstC;
                }
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultBody2(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstC = p.dstC;
            size_t dstCF1 = AlignLo(dstC, 1 * F);
            size_t dstCF2 = AlignLo(dstC, 2 * F);
            size_t dc = 0;
            for (; dc < dstCF2; dc += 2 * F)
            {
                __m512 sums[2][2];
                __m512 bias0 = bias ? _mm512_loadu_ps(bias + dc + 0 * F) : _mm512_setzero_ps();
                __m512 bias1 = bias ? _mm512_loadu_ps(bias + dc + 1 * F) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[0][1] = bias1;
                sums[1][0] = bias0;
                sums[1][1] = bias1;
                KernelHwcDefaultBody2x2(src, p, weight + dc, sums);
                _mm512_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                __m512 sums[2][1];
                __m512 bias0 = bias ? _mm512_loadu_ps(bias + dc) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                KernelHwcDefaultBody2x1(src, p, weight + dc, sums);
                _mm512_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm512_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
            }
            if (dc < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF1);
                __m512 sums[2][1];
                __m512 bias0 = bias ? _mm512_maskz_loadu_ps(tail, bias + dc) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                KernelHwcDefaultBody2x1(src, p, weight + dc, sums);
                _mm512_mask_storeu_ps(dst + dc + 0 * dstC, tail, Activate<type>(sums[0][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 1 * dstC, tail, Activate<type>(sums[1][0], params, dc, tail));
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody6x2(const float * src, const ConvParam32f & p, const float * weight, __m512 sums[6][2])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            __m512 w0, w1, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm512_loadu_ps(weight + 0 * F);
                    w1 = _mm512_loadu_ps(weight + 1 * F);
                    s0 = _mm512_set1_ps(src0[offset]);
                    sums[0][0] = _mm512_fmadd_ps(s0, w0, sums[0][0]);
                    sums[0][1] = _mm512_fmadd_ps(s0, w1, sums[0][1]);
                    s0 = _mm512_set1_ps(src1[offset]);
                    sums[1][0] = _mm512_fmadd_ps(s0, w0, sums[1][0]);
                    sums[1][1] = _mm512_fmadd_ps(s0, w1, sums[1][1]);
                    s0 = _mm512_set1_ps(src2[offset]);
                    sums[2][0] = _mm512_fmadd_ps(s0, w0, sums[2][0]);
                    sums[2][1] = _mm512_fmadd_ps(s0, w1, sums[2][1]);
                    s0 = _mm512_set1_ps(src3[offset]);
                    sums[3][0] = _mm512_fmadd_ps(s0, w0, sums[3][0]);
                    sums[3][1] = _mm512_fmadd_ps(s0, w1, sums[3][1]);
                    s0 = _mm512_set1_ps(src4[offset]);
                    sums[4][0] = _mm512_fmadd_ps(s0, w0, sums[4][0]);
                    sums[4][1] = _mm512_fmadd_ps(s0, w1, sums[4][1]);
                    s0 = _mm512_set1_ps(src5[offset]);
                    sums[5][0] = _mm512_fmadd_ps(s0, w0, sums[5][0]);
                    sums[5][1] = _mm512_fmadd_ps(s0, w1, sums[5][1]);
                    weight += dstC;
                }
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody6x1(const float * src, const ConvParam32f & p, const float * weight, __m512 sums[6][1], __mmask16 tail = -1)
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            __m512 w0, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm512_maskz_loadu_ps(tail, weight + 0 * F);
                    s0 = _mm512_set1_ps(src0[offset]);
                    sums[0][0] = _mm512_fmadd_ps(s0, w0, sums[0][0]);
                    s0 = _mm512_set1_ps(src1[offset]);
                    sums[1][0] = _mm512_fmadd_ps(s0, w0, sums[1][0]);
                    s0 = _mm512_set1_ps(src2[offset]);
                    sums[2][0] = _mm512_fmadd_ps(s0, w0, sums[2][0]);
                    s0 = _mm512_set1_ps(src3[offset]);
                    sums[3][0] = _mm512_fmadd_ps(s0, w0, sums[3][0]);
                    s0 = _mm512_set1_ps(src4[offset]);
                    sums[4][0] = _mm512_fmadd_ps(s0, w0, sums[4][0]);
                    s0 = _mm512_set1_ps(src5[offset]);
                    sums[5][0] = _mm512_fmadd_ps(s0, w0, sums[5][0]);
                    weight += dstC;
                }
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultBody6(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstC = p.dstC;
            size_t dstCF1 = AlignLo(dstC, 1 * F);
            size_t dstCF2 = AlignLo(dstC, 2 * F);
            size_t dc = 0;
            for (; dc < dstCF2; dc += 2 * F)
            {
                __m512 sums[6][2];
                __m512 bias0 = bias ? _mm512_loadu_ps(bias + dc + 0 * F) : _mm512_setzero_ps();
                __m512 bias1 = bias ? _mm512_loadu_ps(bias + dc + 1 * F) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[0][1] = bias1;
                sums[1][0] = bias0;
                sums[1][1] = bias1;
                sums[2][0] = bias0;
                sums[2][1] = bias1;
                sums[3][0] = bias0;
                sums[3][1] = bias1;
                sums[4][0] = bias0;
                sums[4][1] = bias1;
                sums[5][0] = bias0;
                sums[5][1] = bias1;
                KernelHwcDefaultBody6x2(src, p, weight + dc, sums);
                _mm512_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 2 * dstC + 0 * F, Activate<type>(sums[2][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 2 * dstC + 1 * F, Activate<type>(sums[2][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 3 * dstC + 0 * F, Activate<type>(sums[3][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 3 * dstC + 1 * F, Activate<type>(sums[3][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 4 * dstC + 0 * F, Activate<type>(sums[4][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 4 * dstC + 1 * F, Activate<type>(sums[4][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 5 * dstC + 0 * F, Activate<type>(sums[5][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 5 * dstC + 1 * F, Activate<type>(sums[5][1], params, dc + 1 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                __m512 sums[6][1];
                __m512 bias0 = bias ? _mm512_loadu_ps(bias + dc) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                KernelHwcDefaultBody6x1(src, p, weight + dc, sums);
                _mm512_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm512_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
                _mm512_storeu_ps(dst + dc + 2 * dstC, Activate<type>(sums[2][0], params, dc));
                _mm512_storeu_ps(dst + dc + 3 * dstC, Activate<type>(sums[3][0], params, dc));
                _mm512_storeu_ps(dst + dc + 4 * dstC, Activate<type>(sums[4][0], params, dc));
                _mm512_storeu_ps(dst + dc + 5 * dstC, Activate<type>(sums[5][0], params, dc));
            }
            if (dc < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF1);
                __m512 sums[6][1];
                __m512 bias0 = bias ? _mm512_maskz_loadu_ps(tail, bias + dc) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                KernelHwcDefaultBody6x1(src, p, weight + dc, sums, tail);
                _mm512_mask_storeu_ps(dst + dc + 0 * dstC, tail, Activate<type>(sums[0][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 1 * dstC, tail, Activate<type>(sums[1][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 2 * dstC, tail, Activate<type>(sums[2][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 3 * dstC, tail, Activate<type>(sums[3][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 4 * dstC, tail, Activate<type>(sums[4][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 5 * dstC, tail, Activate<type>(sums[5][0], params, dc, tail));
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody8x3(const float * src, const ConvParam32f & p, const float * weight, __m512 sums[8][3])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            const float * src6 = src + 6 * step;
            const float * src7 = src + 7 * step;
            __m512 w0, w1, w2, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm512_loadu_ps(weight + 0 * F);
                    w1 = _mm512_loadu_ps(weight + 1 * F);
                    w2 = _mm512_loadu_ps(weight + 2 * F);
                    s0 = _mm512_set1_ps(src0[offset]);
                    sums[0][0] = _mm512_fmadd_ps(s0, w0, sums[0][0]);
                    sums[0][1] = _mm512_fmadd_ps(s0, w1, sums[0][1]);
                    sums[0][2] = _mm512_fmadd_ps(s0, w2, sums[0][2]);
                    s0 = _mm512_set1_ps(src1[offset]);
                    sums[1][0] = _mm512_fmadd_ps(s0, w0, sums[1][0]);
                    sums[1][1] = _mm512_fmadd_ps(s0, w1, sums[1][1]);
                    sums[1][2] = _mm512_fmadd_ps(s0, w2, sums[1][2]);
                    s0 = _mm512_set1_ps(src2[offset]);
                    sums[2][0] = _mm512_fmadd_ps(s0, w0, sums[2][0]);
                    sums[2][1] = _mm512_fmadd_ps(s0, w1, sums[2][1]);
                    sums[2][2] = _mm512_fmadd_ps(s0, w2, sums[2][2]);
                    s0 = _mm512_set1_ps(src3[offset]);
                    sums[3][0] = _mm512_fmadd_ps(s0, w0, sums[3][0]);
                    sums[3][1] = _mm512_fmadd_ps(s0, w1, sums[3][1]);
                    sums[3][2] = _mm512_fmadd_ps(s0, w2, sums[3][2]);
                    s0 = _mm512_set1_ps(src4[offset]);
                    sums[4][0] = _mm512_fmadd_ps(s0, w0, sums[4][0]);
                    sums[4][1] = _mm512_fmadd_ps(s0, w1, sums[4][1]);
                    sums[4][2] = _mm512_fmadd_ps(s0, w2, sums[4][2]);
                    s0 = _mm512_set1_ps(src5[offset]);
                    sums[5][0] = _mm512_fmadd_ps(s0, w0, sums[5][0]);
                    sums[5][1] = _mm512_fmadd_ps(s0, w1, sums[5][1]);
                    sums[5][2] = _mm512_fmadd_ps(s0, w2, sums[5][2]);
                    s0 = _mm512_set1_ps(src6[offset]);
                    sums[6][0] = _mm512_fmadd_ps(s0, w0, sums[6][0]);
                    sums[6][1] = _mm512_fmadd_ps(s0, w1, sums[6][1]);
                    sums[6][2] = _mm512_fmadd_ps(s0, w2, sums[6][2]);
                    s0 = _mm512_set1_ps(src7[offset]);
                    sums[7][0] = _mm512_fmadd_ps(s0, w0, sums[7][0]);
                    sums[7][1] = _mm512_fmadd_ps(s0, w1, sums[7][1]);
                    sums[7][2] = _mm512_fmadd_ps(s0, w2, sums[7][2]);
                    weight += dstC;
                }
            }
        }

        SIMD_INLINE void KernelHwcDefaultBody8x1(const float * src, const ConvParam32f & p, const float * weight, __m512 sums[8][1])
        {
            size_t size = p.kernelX * p.srcC, dstC = p.dstC, stride = p.srcW * p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            const float * src6 = src + 6 * step;
            const float * src7 = src + 7 * step;
            __m512 w0, s0;
            for (size_t ky = 0; ky < p.kernelY; ++ky)
            {
                size_t offset = ky * stride;
                for (size_t end = offset + size; offset < end; ++offset)
                {
                    w0 = _mm512_loadu_ps(weight + 0 * F);
                    s0 = _mm512_set1_ps(src0[offset]);
                    sums[0][0] = _mm512_fmadd_ps(s0, w0, sums[0][0]);
                    s0 = _mm512_set1_ps(src1[offset]);
                    sums[1][0] = _mm512_fmadd_ps(s0, w0, sums[1][0]);
                    s0 = _mm512_set1_ps(src2[offset]);
                    sums[2][0] = _mm512_fmadd_ps(s0, w0, sums[2][0]);
                    s0 = _mm512_set1_ps(src3[offset]);
                    sums[3][0] = _mm512_fmadd_ps(s0, w0, sums[3][0]);
                    s0 = _mm512_set1_ps(src4[offset]);
                    sums[4][0] = _mm512_fmadd_ps(s0, w0, sums[4][0]);
                    s0 = _mm512_set1_ps(src5[offset]);
                    sums[5][0] = _mm512_fmadd_ps(s0, w0, sums[5][0]);
                    s0 = _mm512_set1_ps(src6[offset]);
                    sums[6][0] = _mm512_fmadd_ps(s0, w0, sums[6][0]);
                    s0 = _mm512_set1_ps(src7[offset]);
                    sums[7][0] = _mm512_fmadd_ps(s0, w0, sums[7][0]);
                    weight += dstC;
                }
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultBody8(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t dstC = p.dstC;
            size_t dstCF1 = AlignLo(dstC, 1 * F);
            size_t dstCF3 = AlignLoAny(dstC, 3 * F);
            size_t dc = 0;
            for (; dc < dstCF3; dc += 3 * F)
            {
                __m512 sums[8][3];
                __m512 bias0 = bias ? _mm512_loadu_ps(bias + dc + 0 * F) : _mm512_setzero_ps();
                __m512 bias1 = bias ? _mm512_loadu_ps(bias + dc + 1 * F) : _mm512_setzero_ps();
                __m512 bias2 = bias ? _mm512_loadu_ps(bias + dc + 2 * F) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[0][1] = bias1;
                sums[0][2] = bias2;
                sums[1][0] = bias0;
                sums[1][1] = bias1;
                sums[1][2] = bias2;
                sums[2][0] = bias0;
                sums[2][1] = bias1;
                sums[2][2] = bias2;
                sums[3][0] = bias0;
                sums[3][1] = bias1;
                sums[3][2] = bias2;
                sums[4][0] = bias0;
                sums[4][1] = bias1;
                sums[4][2] = bias2;
                sums[5][0] = bias0;
                sums[5][1] = bias1;
                sums[5][2] = bias2;
                sums[6][0] = bias0;
                sums[6][1] = bias1;
                sums[6][2] = bias2;
                sums[7][0] = bias0;
                sums[7][1] = bias1;
                sums[7][2] = bias2;
                KernelHwcDefaultBody8x3(src, p, weight + dc, sums);
                _mm512_storeu_ps(dst + dc + 0 * dstC + 0 * F, Activate<type>(sums[0][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 0 * dstC + 1 * F, Activate<type>(sums[0][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 0 * dstC + 2 * F, Activate<type>(sums[0][2], params, dc + 2 * F));
                _mm512_storeu_ps(dst + dc + 1 * dstC + 0 * F, Activate<type>(sums[1][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 1 * dstC + 1 * F, Activate<type>(sums[1][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 1 * dstC + 2 * F, Activate<type>(sums[1][2], params, dc + 2 * F));
                _mm512_storeu_ps(dst + dc + 2 * dstC + 0 * F, Activate<type>(sums[2][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 2 * dstC + 1 * F, Activate<type>(sums[2][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 2 * dstC + 2 * F, Activate<type>(sums[2][2], params, dc + 2 * F));
                _mm512_storeu_ps(dst + dc + 3 * dstC + 0 * F, Activate<type>(sums[3][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 3 * dstC + 1 * F, Activate<type>(sums[3][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 3 * dstC + 2 * F, Activate<type>(sums[3][2], params, dc + 2 * F));
                _mm512_storeu_ps(dst + dc + 4 * dstC + 0 * F, Activate<type>(sums[4][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 4 * dstC + 1 * F, Activate<type>(sums[4][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 4 * dstC + 2 * F, Activate<type>(sums[4][2], params, dc + 2 * F));
                _mm512_storeu_ps(dst + dc + 5 * dstC + 0 * F, Activate<type>(sums[5][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 5 * dstC + 1 * F, Activate<type>(sums[5][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 5 * dstC + 2 * F, Activate<type>(sums[5][2], params, dc + 2 * F));
                _mm512_storeu_ps(dst + dc + 6 * dstC + 0 * F, Activate<type>(sums[6][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 6 * dstC + 1 * F, Activate<type>(sums[6][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 6 * dstC + 2 * F, Activate<type>(sums[6][2], params, dc + 2 * F));
                _mm512_storeu_ps(dst + dc + 7 * dstC + 0 * F, Activate<type>(sums[7][0], params, dc + 0 * F));
                _mm512_storeu_ps(dst + dc + 7 * dstC + 1 * F, Activate<type>(sums[7][1], params, dc + 1 * F));
                _mm512_storeu_ps(dst + dc + 7 * dstC + 2 * F, Activate<type>(sums[7][2], params, dc + 2 * F));
            }
            for (; dc < dstCF1; dc += 1 * F)
            {
                __m512 sums[8][1];
                __m512 bias0 = bias ? _mm512_loadu_ps(bias + dc) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                sums[6][0] = bias0;
                sums[7][0] = bias0;
                KernelHwcDefaultBody6x1(src, p, weight + dc, sums);
                _mm512_storeu_ps(dst + dc + 0 * dstC, Activate<type>(sums[0][0], params, dc));
                _mm512_storeu_ps(dst + dc + 1 * dstC, Activate<type>(sums[1][0], params, dc));
                _mm512_storeu_ps(dst + dc + 2 * dstC, Activate<type>(sums[2][0], params, dc));
                _mm512_storeu_ps(dst + dc + 3 * dstC, Activate<type>(sums[3][0], params, dc));
                _mm512_storeu_ps(dst + dc + 4 * dstC, Activate<type>(sums[4][0], params, dc));
                _mm512_storeu_ps(dst + dc + 5 * dstC, Activate<type>(sums[5][0], params, dc));
                _mm512_storeu_ps(dst + dc + 6 * dstC, Activate<type>(sums[6][0], params, dc));
                _mm512_storeu_ps(dst + dc + 7 * dstC, Activate<type>(sums[7][0], params, dc));
            }
            if (dc < dstC)
            {
                __mmask16 tail = TailMask16(dstC - dstCF1);
                __m512 sums[8][1];
                __m512 bias0 = bias ? _mm512_maskz_loadu_ps(tail, bias + dc) : _mm512_setzero_ps();
                sums[0][0] = bias0;
                sums[1][0] = bias0;
                sums[2][0] = bias0;
                sums[3][0] = bias0;
                sums[4][0] = bias0;
                sums[5][0] = bias0;
                sums[6][0] = bias0;
                sums[7][0] = bias0;
                KernelHwcDefaultBody6x1(src, p, weight + dc, sums, tail);
                _mm512_mask_storeu_ps(dst + dc + 0 * dstC, tail, Activate<type>(sums[0][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 1 * dstC, tail, Activate<type>(sums[1][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 2 * dstC, tail, Activate<type>(sums[2][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 3 * dstC, tail, Activate<type>(sums[3][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 4 * dstC, tail, Activate<type>(sums[4][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 5 * dstC, tail, Activate<type>(sums[5][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 6 * dstC, tail, Activate<type>(sums[6][0], params, dc, tail));
                _mm512_mask_storeu_ps(dst + dc + 7 * dstC, tail, Activate<type>(sums[7][0], params, dc, tail));
            }
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void KernelHwcDefaultBody6_1x1x16(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t size = p.srcC, step = p.srcC * p.strideX;
            const float * src0 = src + 0 * step;
            const float * src1 = src + 1 * step;
            const float * src2 = src + 2 * step;
            const float * src3 = src + 3 * step;
            const float * src4 = src + 4 * step;
            const float * src5 = src + 5 * step;
            __m512 w0, w1, s0, s1;
            __m512 sums[6];
            __m512 bias0 = bias ? _mm512_loadu_ps(bias) : _mm512_setzero_ps();
            sums[0] = bias0;
            sums[1] = bias0;
            sums[2] = bias0;
            sums[3] = bias0;
            sums[4] = bias0;
            sums[5] = bias0;
            size_t offset = 0, size2 = size & (~1);
            for (; offset < size2; offset += 2)
            {
                w0 = _mm512_loadu_ps(weight + 0 * F);
                w1 = _mm512_loadu_ps(weight + 1 * F);
                s0 = _mm512_set1_ps(src0[offset + 0]);
                s1 = _mm512_set1_ps(src1[offset + 0]);
                sums[0] = _mm512_fmadd_ps(s0, w0, sums[0]);
                sums[1] = _mm512_fmadd_ps(s1, w0, sums[1]);
                s0 = _mm512_set1_ps(src0[offset + 1]);
                s1 = _mm512_set1_ps(src1[offset + 1]);
                sums[0] = _mm512_fmadd_ps(s0, w1, sums[0]);
                sums[1] = _mm512_fmadd_ps(s1, w1, sums[1]);
                s0 = _mm512_set1_ps(src2[offset + 0]);
                s1 = _mm512_set1_ps(src3[offset + 0]);
                sums[2] = _mm512_fmadd_ps(s0, w0, sums[2]);
                sums[3] = _mm512_fmadd_ps(s1, w0, sums[3]);
                s0 = _mm512_set1_ps(src2[offset + 1]);
                s1 = _mm512_set1_ps(src3[offset + 1]);
                sums[2] = _mm512_fmadd_ps(s0, w1, sums[2]);
                sums[3] = _mm512_fmadd_ps(s1, w1, sums[3]);
                s0 = _mm512_set1_ps(src4[offset + 0]);
                s1 = _mm512_set1_ps(src5[offset + 0]);
                sums[4] = _mm512_fmadd_ps(s0, w0, sums[4]);
                sums[5] = _mm512_fmadd_ps(s1, w0, sums[5]);
                s0 = _mm512_set1_ps(src4[offset + 1]);
                s1 = _mm512_set1_ps(src5[offset + 1]);
                sums[4] = _mm512_fmadd_ps(s0, w1, sums[4]);
                sums[5] = _mm512_fmadd_ps(s1, w1, sums[5]);
                weight += 2 * F;
            }
            for (; offset < size; ++offset)
            {
                w0 = _mm512_loadu_ps(weight + 0 * F);
                s0 = _mm512_set1_ps(src0[offset]);
                s1 = _mm512_set1_ps(src1[offset]);
                sums[0] = _mm512_fmadd_ps(s0, w0, sums[0]);
                sums[1] = _mm512_fmadd_ps(s1, w0, sums[1]);
                s0 = _mm512_set1_ps(src2[offset]);
                s1 = _mm512_set1_ps(src3[offset]);
                sums[2] = _mm512_fmadd_ps(s0, w0, sums[2]);
                sums[3] = _mm512_fmadd_ps(s1, w0, sums[3]);
                s0 = _mm512_set1_ps(src4[offset]);
                s1 = _mm512_set1_ps(src5[offset]);
                sums[4] = _mm512_fmadd_ps(s0, w0, sums[4]);
                sums[5] = _mm512_fmadd_ps(s1, w0, sums[5]);
                weight += F;
            }
            _mm512_storeu_ps(dst + 0 * F, Activate<type>(sums[0], params, 0));
            _mm512_storeu_ps(dst + 1 * F, Activate<type>(sums[1], params, 0));
            _mm512_storeu_ps(dst + 2 * F, Activate<type>(sums[2], params, 0));
            _mm512_storeu_ps(dst + 3 * F, Activate<type>(sums[3], params, 0));
            _mm512_storeu_ps(dst + 4 * F, Activate<type>(sums[4], params, 0));
            _mm512_storeu_ps(dst + 5 * F, Activate<type>(sums[5], params, 0));
        }

        template<::SimdConvolutionActivationType type> void ConvolutionDirectNhwcConvolutionBiasActivationDefault(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            bool is1x1x16 = p.dstC == 16 && p.kernelX == 1 && p.kernelY == 1;
            size_t noseH = p.padY, noseW = p.padX;
            size_t bodyH = p.srcH - p.kernelY + 1 + noseH, bodyW = p.srcW - p.kernelX + 1 + noseW;
            size_t tailH = bodyH + p.padH, tailW = bodyW + p.padW;
            size_t bodyW2 = AlignLoAny(bodyW - noseW, 2 * p.strideX) + noseW;
            size_t bodyW6 = AlignLoAny(bodyW - noseW, 6 * p.strideX) + noseW;
            size_t bodyW8 = AlignLoAny(bodyW - noseW, 8 * p.strideX) + noseW;
            size_t wS = p.srcC*p.dstC;
            size_t kY = p.kernelY - noseH, kX = p.kernelX - noseW, kH = bodyH + p.kernelY - 1, kW = bodyW + p.kernelX - 1;
            size_t sy = 0;
            for (; sy < noseH; sy += p.strideY)
            {
                size_t sx = 0;
                const float * w = weight + (noseH - sy) * p.kernelY * wS;
                for (; sx < noseW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src, p, kY + sy, kX + sx, w + (noseW - sx)*wS, bias, params, dst);
                for (; sx < bodyW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, kY + sy, p.kernelX, w, bias, params, dst);
                for (; sx < tailW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, kY + sy, kW - sx, w, bias, params, dst);
            }
            src += (sy - noseH)*p.srcW*p.srcC;
            for (; sy < bodyH; sy += p.strideY)
            {
                size_t sx = 0;
                for (; sx < noseW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src, p, p.kernelY, kX + sx, weight + (noseW - sx)*wS, bias, params, dst);
                if (is1x1x16)
                {
                    for (; sx < bodyW6; sx += 6 * p.strideX, dst += 6 * p.dstC)
                        KernelHwcDefaultBody6_1x1x16<type>(src + (sx - noseW) * p.srcC, p, weight, bias, params, dst);
                }
                else if (p.dstC == 48)
                {
                    for (; sx < bodyW8; sx += 8 * p.strideX, dst += 8 * p.dstC)
                        KernelHwcDefaultBody8<type>(src + (sx - noseW) * p.srcC, p, weight, bias, params, dst);
                }
                else
                {
                    for (; sx < bodyW6; sx += 6 * p.strideX, dst += 6 * p.dstC)
                        KernelHwcDefaultBody6<type>(src + (sx - noseW) * p.srcC, p, weight, bias, params, dst);
                }
                for (; sx < bodyW2; sx += 2 * p.strideX, dst += 2 * p.dstC)
                    KernelHwcDefaultBody2<type>(src + (sx - noseW) * p.srcC, p, weight, bias, params, dst);
                for (; sx < bodyW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, p.kernelY, p.kernelX, weight, bias, params, dst);
                for (; sx < tailW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, p.kernelY, kW - sx, weight, bias, params, dst);
                src += p.strideY*p.srcW*p.srcC;
            }
            for (; sy < tailH; sy += p.strideY)
            {
                size_t sx = 0;
                for (; sx < noseW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src, p, kH - sy, kX + sx, weight + (noseW - sx)*wS, bias, params, dst);
                for (; sx < bodyW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, kH - sy, p.kernelX, weight, bias, params, dst);
                for (; sx < tailW; sx += p.strideX, dst += p.dstC)
                    KernelHwcDefaultEdge<type>(src + (sx - noseW) * p.srcC, p, kH - sy, kW - sx, weight, bias, params, dst);
                src += p.strideY*p.srcW*p.srcC;
            }
        }

        template<::SimdConvolutionActivationType type> void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F);
            size_t size2F = AlignLo(size, 2 * F);
            size_t size4F = AlignLo(size, 4 * F);
            size_t size8F = AlignLo(size, 8 * F);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < size8F; i += 8 * F)
                    {
                        __m512 sums[8];
                        if (bias)
                        {
                            sums[0] = _mm512_loadu_ps(bias + i + 0 * F);
                            sums[1] = _mm512_loadu_ps(bias + i + 1 * F);
                            sums[2] = _mm512_loadu_ps(bias + i + 2 * F);
                            sums[3] = _mm512_loadu_ps(bias + i + 3 * F);
                            sums[4] = _mm512_loadu_ps(bias + i + 4 * F);
                            sums[5] = _mm512_loadu_ps(bias + i + 5 * F);
                            sums[6] = _mm512_loadu_ps(bias + i + 6 * F);
                            sums[7] = _mm512_loadu_ps(bias + i + 7 * F);
                        }
                        else
                        {
                            sums[0] = _mm512_setzero_ps();
                            sums[1] = _mm512_setzero_ps();
                            sums[2] = _mm512_setzero_ps();
                            sums[3] = _mm512_setzero_ps();
                            sums[4] = _mm512_setzero_ps();
                            sums[5] = _mm512_setzero_ps();
                            sums[6] = _mm512_setzero_ps();
                            sums[7] = _mm512_setzero_ps();
                        }
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)*size + i;
                                        const float * ps = src + (sy*p.srcW + sx)*size + i;
                                        sums[0] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), sums[0]);
                                        sums[1] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), sums[1]);
                                        sums[2] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * F), _mm512_loadu_ps(pw + 2 * F), sums[2]);
                                        sums[3] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * F), _mm512_loadu_ps(pw + 3 * F), sums[3]);
                                        sums[4] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 4 * F), _mm512_loadu_ps(pw + 4 * F), sums[4]);
                                        sums[5] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 5 * F), _mm512_loadu_ps(pw + 5 * F), sums[5]);
                                        sums[6] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 6 * F), _mm512_loadu_ps(pw + 6 * F), sums[6]);
                                        sums[7] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 7 * F), _mm512_loadu_ps(pw + 7 * F), sums[7]);
                                    }
                                }
                            }
                        }
                        _mm512_storeu_ps(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        _mm512_storeu_ps(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                        _mm512_storeu_ps(dst + i + 2 * F, Activate<type>(sums[2], params, i + 2 * F));
                        _mm512_storeu_ps(dst + i + 3 * F, Activate<type>(sums[3], params, i + 3 * F));
                        _mm512_storeu_ps(dst + i + 4 * F, Activate<type>(sums[4], params, i + 4 * F));
                        _mm512_storeu_ps(dst + i + 5 * F, Activate<type>(sums[5], params, i + 5 * F));
                        _mm512_storeu_ps(dst + i + 6 * F, Activate<type>(sums[6], params, i + 6 * F));
                        _mm512_storeu_ps(dst + i + 7 * F, Activate<type>(sums[7], params, i + 7 * F));
                    }
                    for (; i < size4F; i += 4 * F)
                    {
                        __m512 sums[4];
                        if (bias)
                        {
                            sums[0] = _mm512_loadu_ps(bias + i + 0 * F);
                            sums[1] = _mm512_loadu_ps(bias + i + 1 * F);
                            sums[2] = _mm512_loadu_ps(bias + i + 2 * F);
                            sums[3] = _mm512_loadu_ps(bias + i + 3 * F);
                        }
                        else
                        {
                            sums[0] = _mm512_setzero_ps();
                            sums[1] = _mm512_setzero_ps();
                            sums[2] = _mm512_setzero_ps();
                            sums[3] = _mm512_setzero_ps();
                        }
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)*size + i;
                                        const float * ps = src + (sy*p.srcW + sx)*size + i;
                                        sums[0] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), sums[0]);
                                        sums[1] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), sums[1]);
                                        sums[2] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * F), _mm512_loadu_ps(pw + 2 * F), sums[2]);
                                        sums[3] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * F), _mm512_loadu_ps(pw + 3 * F), sums[3]);
                                    }
                                }
                            }
                        }
                        _mm512_storeu_ps(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        _mm512_storeu_ps(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                        _mm512_storeu_ps(dst + i + 2 * F, Activate<type>(sums[2], params, i + 2 * F));
                        _mm512_storeu_ps(dst + i + 3 * F, Activate<type>(sums[3], params, i + 3 * F));
                    }
                    for (; i < size2F; i += 2 * F)
                    {
                        __m512 sums[2];
                        if (bias)
                        {
                            sums[0] = _mm512_loadu_ps(bias + i + 0 * F);
                            sums[1] = _mm512_loadu_ps(bias + i + 1 * F);
                        }
                        else
                        {
                            sums[0] = _mm512_setzero_ps();
                            sums[1] = _mm512_setzero_ps();
                        }
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)*size + i;
                                        const float * ps = src + (sy*p.srcW + sx)*size + i;
                                        sums[0] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), sums[0]);
                                        sums[1] = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), sums[1]);
                                    }
                                }
                            }
                        }
                        _mm512_storeu_ps(dst + i + 0 * F, Activate<type>(sums[0], params, i + 0 * F));
                        _mm512_storeu_ps(dst + i + 1 * F, Activate<type>(sums[1], params, i + 1 * F));
                    }
                    for (; i < size; i += F)
                    {
                        __mmask16 tail = i < sizeF ? __mmask16(-1) : TailMask16(size - i);
                        __m512 sum = bias ? _mm512_maskz_loadu_ps(tail, bias + i) : _mm512_setzero_ps();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)*size + i;
                                        const float * ps = src + (sy*p.srcW + sx)*size + i;
                                        sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps), _mm512_maskz_loadu_ps(tail, pw), sum);
                                    }
                                }
                            }
                        }
                        _mm512_mask_storeu_ps(dst + i, tail, Activate<type>(sum, params, i, tail));
                    }
                    dst += p.dstC;
                }
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge(const float * src, const ConvParam32f & p, size_t dy, size_t dx, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 sum = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float * pw = weight + (ky * 3 + kx) * srcC;
                                const float * ps = src + (sy*p.srcW + sx) * srcC;
                                sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                            }
                        }
                    }
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float * pw = weight + (ky*3 + kx) * srcC;
                                const float * ps = src + (sy*p.srcW + sx) * srcC;
                                sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps), _mm512_maskz_loadu_ps(tail, pw), sum);
                            }
                        }
                    }
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main1(const float * src, size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 sum = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * srcC), _mm512_loadu_ps(pw + 0 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * srcC), _mm512_loadu_ps(pw + 1 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * srcC), _mm512_loadu_ps(pw + 2 * srcC), sum);
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 0 * srcC), _mm512_maskz_loadu_ps(tail, pw + 0 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 1 * srcC), _mm512_maskz_loadu_ps(tail, pw + 1 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 2 * srcC), _mm512_maskz_loadu_ps(tail, pw + 2 * srcC), sum);
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main2(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            __m512 sum0, sum1, w0;
            for (; c < srcCF; c += F)
            {
                sum0 = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 0 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 0 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 1 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 1 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 2 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 2 * srcC), w0, sum1);
                    pw += srcC;
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm512_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
                src += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                sum0 = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 0 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 0 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 1 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 1 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 2 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 2 * srcC), w0, sum1);
                    pw += srcC;
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum0, params, c, tail));
                _mm512_mask_storeu_ps(dst + c + srcC, tail, Activate<type>(sum1, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);                    
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);                    
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                }
                _mm512_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm512_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm512_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm512_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
                src += F;
                dst += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                }
                _mm512_mask_storeu_ps(dst + 0 * srcC, tail, Activate<type>(sum0, params, c, tail));
                _mm512_mask_storeu_ps(dst + 1 * srcC, tail, Activate<type>(sum1, params, c, tail));
                _mm512_mask_storeu_ps(dst + 2 * srcC, tail, Activate<type>(sum2, params, c, tail));
                _mm512_mask_storeu_ps(dst + 3 * srcC, tail, Activate<type>(sum3, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge16(const float * src, const ConvParam32f & p, size_t dy, size_t dx, const __m512 * weight, __m512 bias, const float * params, float * dst)
        {
            __m512 sum = bias;
            for (size_t ky = 0; ky < 3; ++ky)
            {
                size_t sy = dy * p.strideY + ky - p.padY;
                if (sy < p.srcH)
                {
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        if (sx < p.srcW)
                        {
                            const float * ps = src + (sy*p.srcW + sx) * F;
                            sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), weight[ky * 3 + kx], sum);
                        }
                    }
                }
            }
            _mm512_storeu_ps(dst, Activate<type>(sum, params, 0));
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main16x1(const float * src, size_t srcS, const __m512 * weight, __m512 bias, const float * params, float * dst)
        {
            __m512 sum = bias;
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[0], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[1], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[2], sum);
            src += srcS;
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[3], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[4], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[5], sum);
            src += srcS;
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[6], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[7], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[8], sum);
            _mm512_storeu_ps(dst, Activate<type>(sum, params, 0));
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main16x2(const float * src, size_t srcS, const __m512 * weight, __m512 bias, const float * params, float * dst)
        {
            __m512 sum0 = bias;
            __m512 sum1 = bias;
            for (size_t ky = 0; ky < 3; ++ky)
            {
                __m512 s0 = _mm512_loadu_ps(src + 0 * F);
                __m512 s1 = _mm512_loadu_ps(src + 1 * F);
                __m512 s2 = _mm512_loadu_ps(src + 2 * F);
                __m512 s3 = _mm512_loadu_ps(src + 3 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[0], sum0);
                sum1 = _mm512_fmadd_ps(s1, weight[0], sum1);
                sum0 = _mm512_fmadd_ps(s1, weight[1], sum0);
                sum1 = _mm512_fmadd_ps(s2, weight[1], sum1);
                sum0 = _mm512_fmadd_ps(s2, weight[2], sum0);
                sum1 = _mm512_fmadd_ps(s3, weight[2], sum1);
                src += srcS;
                weight += 3;
            }
            _mm512_storeu_ps(dst + 0, Activate<type>(sum0, params, 0));
            _mm512_storeu_ps(dst + F, Activate<type>(sum1, params, 0));
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main48(const float * src, size_t srcS, const __m512 * weight, const float * bias, const float * params, float * dst)
        {
            __m512 sum0, sum1, sum2;
            if (bias)
            {
                sum0 = _mm512_loadu_ps(bias + 0 * F);
                sum1 = _mm512_loadu_ps(bias + 1 * F);
                sum2 = _mm512_loadu_ps(bias + 2 * F);
            }
            else
            {
                sum0 = _mm512_setzero_ps();
                sum1 = _mm512_setzero_ps();
                sum2 = _mm512_setzero_ps();
            }
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[0], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[1], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[2], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 3 * F), weight[3], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 4 * F), weight[4], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 5 * F), weight[5], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 6 * F), weight[6], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 7 * F), weight[7], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 8 * F), weight[8], sum2);
            src += srcS;
            weight += 9;
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[0], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[1], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[2], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 3 * F), weight[3], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 4 * F), weight[4], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 5 * F), weight[5], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 6 * F), weight[6], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 7 * F), weight[7], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 8 * F), weight[8], sum2);
            src += srcS;
            weight += 9;
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[0], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[1], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[2], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 3 * F), weight[3], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 4 * F), weight[4], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 5 * F), weight[5], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 6 * F), weight[6], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 7 * F), weight[7], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 8 * F), weight[8], sum2);
            _mm512_storeu_ps(dst + 0 * F, Activate<type>(sum0, params, 0 * F));
            _mm512_storeu_ps(dst + 1 * F, Activate<type>(sum1, params, 1 * F));
            _mm512_storeu_ps(dst + 2 * F, Activate<type>(sum2, params, 2 * F));
        }

        template<::SimdConvolutionActivationType type> void ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3(const float * src, const ConvParam32f & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcS = p.srcC*p.srcW;
            size_t srcX = p.srcC*p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dstW4 = AlignLo(dstW - p.padX, 4) + p.padX;
            if (p.dstC == F && p.strideX == 1)
            {
                __m512 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm512_loadu_ps(weight + i * F);
                __m512 _bias = bias ? _mm512_loadu_ps(bias) : _mm512_setzero_ps();
                size_t dy = 0;
                for (; dy < p.padY; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge16<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                for (; dy < dstH; ++dy)
                {
                    size_t dx = 0;
                    for (; dx < p.padX; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge16<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                    size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                    for (; dx < dstW2; dx += 2)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main16x2<type>(src + offset, srcS, _weight, _bias, params, dst), offset += 2*F, dst += 2*F;
                    for (; dx < dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main16x1<type>(src + offset, srcS, _weight, _bias, params, dst), offset += F, dst += F;
                    for (; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge16<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                }
                for (; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge16<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
            }
            else
            {
                size_t dy = 0;
                for (; dy < p.padY; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                for (; dy < dstH; ++dy)
                {
                    size_t dx = 0;
                    for (; dx < p.padX; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                    size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                    if (p.srcC == 48)
                    {
                        __m512 _weight[27];
                        for (size_t i = 0; i < 27; ++i)
                            _weight[i] = _mm512_loadu_ps(weight + i * F);
                        for (; dx < dstW; ++dx)
                            ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main48<type>(src + offset, srcS, _weight, bias, params, dst), dst += p.dstC, offset += srcX;
                    }
                    else
                        for (; dx < dstW4; dx += 4)
                            ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main4<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 4 * p.dstC, offset += 4 * srcX;
                    for (; dx < dstW2; dx += 2)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main2<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 2 * p.dstC, offset += 2 * srcX;
                    for (; dx < dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Main1<type>(src + offset, srcS, p.srcC, weight, bias, params, dst), dst += p.dstC, offset += srcX;
                    for (; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                }
                for (; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
            }
        }

        template <::SimdConvolutionActivationType type> SynetConvolution32fDirectNhwc::ConvolutionBiasActivationPtr GetConvolutionBiasActivation(const ConvParam32f & p)
        {
            if (p.group == 1)
                return ConvolutionDirectNhwcConvolutionBiasActivationDefault<type>;
            else if (p.IsDepthwise())
            {
                if(p.IsKernel(3) && p.IsDilation(1))
                    return ConvolutionDirectNhwcConvolutionBiasActivationDepthwise3x3<type>;
                else
                    return ConvolutionDirectNhwcConvolutionBiasActivationDepthwise<type>;
            }
            return NULL;
        }

        SynetConvolution32fDirectNhwc::ConvolutionBiasActivationPtr SynetConvolution32fDirectNhwc::SetConvolutionBiasActivation()
        {
            const ConvParam32f & p = _param;
            SynetConvolution32fDirectNhwc::ConvolutionBiasActivationPtr func = NULL;
            if (p.dstC > HF && p.dstC != 24 && p.dstH >= p.padY + p.padH && p.dstW >= p.padX + p.padW)
            {
                switch (p.activation)
                {
                case ::SimdConvolutionActivationIdentity: func = GetConvolutionBiasActivation<::SimdConvolutionActivationIdentity>(p); break;
                case ::SimdConvolutionActivationRelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationRelu>(p); break;
                case ::SimdConvolutionActivationLeakyRelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationLeakyRelu>(p); break;
                case ::SimdConvolutionActivationRestrictRange: func = GetConvolutionBiasActivation<::SimdConvolutionActivationRestrictRange>(p); break;
                case ::SimdConvolutionActivationPrelu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationPrelu>(p); break;
                case ::SimdConvolutionActivationElu: func = GetConvolutionBiasActivation<::SimdConvolutionActivationElu>(p); break;
                case ::SimdConvolutionActivationHswish: func = GetConvolutionBiasActivation<::SimdConvolutionActivationHswish>(p); break;
                case ::SimdConvolutionActivationMish: func = GetConvolutionBiasActivation<::SimdConvolutionActivationMish>(p); break;
                }
            }
            return func ? func : Avx2::SynetConvolution32fDirectNhwc::SetConvolutionBiasActivation();
        };

        //---------------------------------------------------------------------

        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam32f& p)
            : Avx2::SynetConvolution32fNhwcDirect(p)
        {
            if (p.dstC <= Avx::F)
                return;
#ifdef SIMD_SYNET_CONVOLUTION_NHWC_DIRECT_OLD
            //_old.enable = true;
            if (_old.enable)
            {
                if (Set2f(p, _old.convolution))
                    OldSetAlgParam(F);
            }
            else
#endif
            {
                RunFuncs funcs;
                for (size_t n = 2; n <= 3; ++n)
                {
                    funcs.push_back(RunFunc(Ext() + "-" + ToStr(n)));
                    SetAlgParam(F, n, funcs.back().alg);
                    if (!SetRt(p, funcs.back().alg))
                        return;
                }
                _run.Init(funcs);
            }
        }

        bool SynetConvolution32fNhwcDirect::SetRt(const ConvParam32f& p, AlgParam& a)
        {
            switch (a.microD)
            {
            case 2 * F: return Set2r(p, a);
            case 3 * F: return Set3r(p, a);
            default:
                return false;
            }
        }

        //---------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            ConvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            else if (Avx::SynetConvolution32fDepthwiseDotProduct::Preferable(param))
                return new Avx::SynetConvolution32fDepthwiseDotProduct(param);
            else if (SynetConvolution32fWinograd::Preferable(param))
                return new SynetConvolution32fWinograd(param);
            else if (SynetConvolution32fGemmNT::Preferable(param))
                return new SynetConvolution32fGemmNT(param);
            else if (SynetConvolution32fDirectNchw::Preferable(param))
                return new Avx512f::SynetConvolution32fDirectNchw(param);
            else if (SynetConvolution32fNhwcDirect::Preferable(param))
                return new SynetConvolution32fNhwcDirect(param);
            else if (SynetConvolution32fDirectNhwc::Preferable(param))
                return new SynetConvolution32fDirectNhwc(param);
            else
                return new SynetConvolution32fGemmNN(param);
        }
    }
#endif//SIMD_AVX512F_ENABLE
}

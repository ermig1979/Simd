/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse2
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst)
        {
            size_t aligned = trans ? AlignLo(count, F) : AlignLo(size, F);
            if (activation == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    Sse2::SynetAddBias(bias, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    __m128 _0 = _mm_set1_ps(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 _dst = _mm_loadu_ps(dst + i);
                                __m128 _bias = _mm_loadu_ps(bias + i);
                                _mm_storeu_ps(dst + i, _mm_max_ps(_0, _mm_add_ps(_dst, _bias)));
                            }
                            for (; i < count; ++i)
                                dst[i] = Simd::Max(0.0f, dst[i] + bias[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 _dst = _mm_loadu_ps(dst + j);
                                _mm_storeu_ps(dst + j, _mm_max_ps(_0, _mm_add_ps(_dst, _bias)));
                            }
                            for (; j < size; ++j)
                                dst[j] = Simd::Max(0.0f, dst[j] + bias[i]);
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
                    __m128 _slope = _mm_set1_ps(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(bias + i));
                                _mm_storeu_ps(dst + i, SynetRelu32f(value, _slope));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetRelu32f(dst[i] + bias[i], slope);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _bias);
                                _mm_storeu_ps(dst + j, SynetRelu32f(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetRelu32f(dst[j] + bias[i], slope);
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
                    __m128 _lower = _mm_set1_ps(lower);
                    __m128 _upper = _mm_set1_ps(upper);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(bias + i));
                                _mm_storeu_ps(dst + i, _mm_min_ps(_mm_max_ps(_lower, value), _upper));
                            }
                            for (; i < count; ++i)
                                dst[i] = Simd::RestrictRange(dst[i] + bias[i], lower, upper);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _bias);
                                _mm_storeu_ps(dst + j, _mm_min_ps(_mm_max_ps(_lower, value), _upper));
                            }
                            for (; j < size; ++j)
                                dst[j] = Simd::RestrictRange(dst[j] + bias[i], lower, upper);
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
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(bias + i));
                                _mm_storeu_ps(dst + i, SynetRelu32f(value, _mm_loadu_ps(params + i)));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetRelu32f(dst[i] + bias[i], params[i]);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            __m128 _slope = _mm_set1_ps(params[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _bias);
                                _mm_storeu_ps(dst + j, SynetRelu32f(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetRelu32f(dst[j] + bias[i], params[i]);
                            dst += size;
                        }
                    }
                }
                else
                    SynetPreluLayerForward(dst, params, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationElu)
            {
                float alpha = params[0];
                if (bias)
                {
                    __m128 _alpha = _mm_set1_ps(alpha);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + i), _mm_loadu_ps(bias + i));
                                _mm_storeu_ps(dst + i, Sse2::Elu(value, _alpha));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetElu32f(dst[i] + bias[i], alpha);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(_mm_loadu_ps(dst + j), _bias);
                                _mm_storeu_ps(dst + j, Sse2::Elu(value, _alpha));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetElu32f(dst[j] + bias[i], alpha);
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
                    __m128 _shift = _mm_set1_ps(shift);
                    __m128 _scale = _mm_set1_ps(scale);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, SynetHswish32f(value, _shift, _scale));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetHswish32f(dst[i] + bias[i], shift, scale);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, SynetHswish32f(value, _shift, _scale));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetHswish32f(dst[j] + bias[i], shift, scale);
                            dst += size;
                        }
                    }
                }
                else
                    SynetHswish32f(dst, count * size, &shift, &scale, dst);
            }
            else if (activation == ::SimdConvolutionActivationMish)
            {
                float threshold = params[0];
                if (bias)
                {
                    __m128 _threshold = _mm_set1_ps(threshold);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Mish(value, _threshold));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetMish32f(dst[i] + bias[i], threshold);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Mish(value, _threshold));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetMish32f(dst[j] + bias[i], threshold);
                            dst += size;
                        }
                    }
                }
                else
                    SynetMish32f(dst, size* count, &threshold, dst);
            }
            else if (activation == ::SimdConvolutionActivationHardSigmoid)
            {
                float scale = params[0];
                float shift = params[1];
                if (bias)
                {
                    __m128 _scale = _mm_set1_ps(scale);
                    __m128 _shift = _mm_set1_ps(shift);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, SynetHardSigmoid32f(value, _scale, _shift));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetHardSigmoid32f(dst[i] + bias[i], scale, shift);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, SynetHardSigmoid32f(value, _scale, _shift));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetHardSigmoid32f(dst[j] + bias[i], scale, shift);
                            dst += size;
                        }
                    }
                }
                else
                    SynetHardSigmoid32f(dst, count * size, &scale, &shift, dst);
            }
            else if (activation == ::SimdConvolutionActivationSwish)
            {
                float slope = params[0];
                if (bias)
                {
                    __m128 _slope = _mm_set1_ps(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Swish(value, _slope));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetSwish32f(dst[i] + bias[i], slope);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m128 value = _mm_add_ps(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Swish(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetSwish32f(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    SynetSwish32f(dst, count * size, &slope, dst);
            }
            else
            {
                Base::ConvolutionBiasAndActivation(bias, count, size, activation, params, trans, dst);
            }
        }

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam32f & p)
            : Base::SynetConvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Sse2::Gemm32fNN, "Sse2", p.gemm, "Ext"));
            if (_param.trans && _param.group == 1)
            {
                if (GemmRuntime())
                {
                    _gemmCb.Init(InitGemmCbFuncs(Sse2::Gemm32fNNcbBufferSize, Sse2::Gemm32fNNcbReorderB, Sse2::Gemm32fNNcbRun, "Sse2", GemmKernelF2, GemmKernelF3));
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M*_merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Sse2::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Sse2::Gemm32fNNcbRun;
                _nhwcReorderB = Sse2::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Sse2::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam32f & p)
            : Base::SynetConvolution32fWinograd(p)
        {
            if (p.kernelY == 1 && p.kernelX == 3)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Sse2::WinogradKernel1x3Block1x4SetFilter;
                    _setInput = Sse2::WinogradKernel1x3Block1x4SetInput;
                    _setOutput = Sse2::WinogradKernel1x3Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 1 && p.kernelX == 5)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Sse2::WinogradKernel1x5Block1x4SetFilter;
                    _setInput = Sse2::WinogradKernel1x5Block1x4SetInput;
                    _setOutput = Sse2::WinogradKernel1x5Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 2 && p.kernelX == 2)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    SetBlock(4, 4);
                    _setFilter = Sse2::WinogradKernel2x2Block4x4SetFilter;
                    _setInput = Sse2::WinogradKernel2x2Block4x4SetInput;
                    _setOutput = Sse2::WinogradKernel2x2Block4x4SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    SetBlock(2, 2);
                    _setFilter = Sse2::WinogradKernel2x2Block2x2SetFilter;
                    _setInput = Sse2::WinogradKernel2x2Block2x2SetInput;
                    _setOutput = Sse2::WinogradKernel2x2Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else if (p.kernelY == 3 && p.kernelX == 3)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    _setFilter = Sse2::WinogradKernel3x3Block4x4SetFilter;
                    _setInput = Sse2::WinogradKernel3x3Block4x4SetInput;
                    _setOutput = Sse2::WinogradKernel3x3Block4x4SetOutput;
                }
                else if (_blockY == 3 && _blockX == 3)
                {
                    _setFilter = Sse2::WinogradKernel3x3Block3x3SetFilter;
                    _setInput = Sse2::WinogradKernel3x3Block3x3SetInput;
                    _setOutput = Sse2::WinogradKernel3x3Block3x3SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    _setFilter = Sse2::WinogradKernel3x3Block2x2SetFilter;
                    _setInput = Sse2::WinogradKernel3x3Block2x2SetInput;
                    _setOutput = Sse2::WinogradKernel3x3Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else
                assert(0);
            _gemm.Init(InitGemmFuncs(Sse2::Gemm32fNN, "Sse2", p.gemm, "Ext"));
            if (_param.trans)
            {
                if (NHWC_GEMM_RUNTIME)
                {
                    _gemmCb.Init(InitGemmCbFuncs(Sse2::Gemm32fNNcbBufferSize, Sse2::Gemm32fNNcbReorderB, Sse2::Gemm32fNNcbRun, "Sse2", GemmKernelF2, GemmKernelF3));
                    _nhwcStrideW = _gemmCb.At(0).BufferSize(_M*_merge, _N, _K);
                }
                else
                    _nhwcStrideW = Sse2::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                _nhwcWeight.Resize(_nhwcStrideW*_count);
                _nhwcRun = Sse2::Gemm32fNNcbRun;
                _nhwcReorderB = Sse2::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Sse2::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fDepthwiseDotProduct::SynetConvolution32fDepthwiseDotProduct(const ConvParam32f & p)
            : Base::SynetConvolution32fDepthwiseDotProduct(p)
        {
        }

        SIMD_INLINE void DotProduct(const float * a, const float * b, size_t offset, __m128 & sum)
        {
            __m128 _a = _mm_loadu_ps(a + offset);
            __m128 _b = _mm_loadu_ps(b + offset);
            sum = _mm_add_ps(_mm_mul_ps(_a, _b), sum);
        }

        SIMD_INLINE float DotProduct(const float * a, const float * b, size_t size)
        {
            float sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        DotProduct(a, b, i + F * 0, sums[0]);
                        DotProduct(a, b, i + F * 1, sums[1]);
                        DotProduct(a, b, i + F * 2, sums[2]);
                        DotProduct(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += F)
                    DotProduct(a, b, i, sums[0]);
                sum += ExtractSum(sums[0]);
            }
            for (; i < size; ++i)
                sum += a[i] * b[i];
            return sum;
        }

        void SynetConvolution32fDepthwiseDotProduct::Forward(const float * src, float * buf, float * dst)
        {
            for (size_t b = 0; b < _batch; ++b)
            {
                if (_bias)
                {
                    for (size_t i = 0; i < _count; ++i)
                        dst[i] = DotProduct(src + i * _size, _weight + i * _size, _size) + _bias[i];
                }
                else
                {
                    for (size_t i = 0; i < _count; ++i)
                        dst[i] = DotProduct(src + i * _size, _weight + i * _size, _size);
                }
                if (_param.activation)
                    ConvolutionBiasAndActivation(NULL, _count, 1, _param.activation, _params, ::SimdFalse, dst);
                src += _sizeS;
                dst += _sizeD;
            }
        }

        //---------------------------------------------------------------------

        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam32f& p)
            : Base::SynetConvolution32fNhwcDirect(p)
        {
            //_old.enable = true;
            if (_old.enable)
            {
                if (Set2f(p, _old.convolution))
                    OldSetAlgParam(F);
            }
            else
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

        bool SynetConvolution32fNhwcDirect::Preferable(const ConvParam32f& p)
        {
            if (p.trans != SimdTrue || p.group != 1)
                return false;
            if (!p.Is1x1() && p.dstW < 6 + p.padX + p.padW)
                return false;
            if (p.Is1x1() && (p.srcC >= 2 * p.dstC || (p.activation == SimdConvolutionActivationIdentity && p.srcC > 512) || p.srcC > 512) && 
                p.dstH*p.dstW < p.srcC * p.dstC * 10)
                return false;
            if (p.kernelY > p.srcH || p.kernelX > p.srcW)
                return false;
            if ((p.strideY > 1 && p.strideX > 1) && p.srcC > 32 && float(p.kernelY * p.kernelX) / float(p.strideY * p.strideX) < 3.0f)
                return false;
            if ((p.padX + p.padW)*3.0f > float(p.srcW))
                return false;
            return true;
        }

        //---------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            ConvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            else if (SynetConvolution32fDepthwiseDotProduct::Preferable(param))
                return new SynetConvolution32fDepthwiseDotProduct(param);
            else if (SynetConvolution32fWinograd::Preferable(param))
                return new SynetConvolution32fWinograd(param);
            else if (SynetConvolution32fDirectNchw::Preferable(param))
                return new SynetConvolution32fDirectNchw(param);
            else if (SynetConvolution32fNhwcDirect::Preferable(param))
                return new SynetConvolution32fNhwcDirect(param);
            else if (SynetConvolution32fDirectNhwc::Preferable(param))
                return new SynetConvolution32fDirectNhwc(param);
            else
                return new SynetConvolution32fGemmNN(param);
        }
    }
#endif//SIMD_SSE2_ENABLE
}

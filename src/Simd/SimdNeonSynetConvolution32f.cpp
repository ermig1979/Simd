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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Neon
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst)
        {
            size_t aligned = trans ? AlignLo(count, F) : AlignLo(size, F);
            if (activation == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    Neon::SynetAddBias(bias, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    float32x4_t _0 = vdupq_n_f32(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t _dst = Load<false>(dst + i);
                                float32x4_t _bias = Load<false>(bias + i);
                                Store<false>(dst + i, vmaxq_f32(_0, vaddq_f32(_dst, _bias)));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t _dst = Load<false>(dst + j);
                                Store<false>(dst + j, vmaxq_f32(_0, vaddq_f32(_dst, _bias)));
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
                    Neon::SynetRelu32f(dst, size*count, &slope, dst);
                }
            }
            else if (activation == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    float32x4_t _0 = vdupq_n_f32(0.0f);
                    float32x4_t _slope = vdupq_n_f32(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, SynetRelu32f(value, _slope, _0));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, SynetRelu32f(value, _slope, _0));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetRelu32f(dst[j] + bias[i], slope);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetRelu32f(dst, size*count, &slope, dst);
            }
            else if (activation == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    float32x4_t _lower = vdupq_n_f32(lower);
                    float32x4_t _upper = vdupq_n_f32(upper);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, vminq_f32(vmaxq_f32(_lower, value), _upper));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, vminq_f32(vmaxq_f32(_lower, value), _upper));
                            }
                            for (; j < size; ++j)
                                dst[j] = Simd::RestrictRange(dst[j] + bias[i], lower, upper);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetRestrictRange32f(dst, size*count, &lower, &upper, dst);
            }
            else if (activation == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    float32x4_t _0 = vdupq_n_f32(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, SynetRelu32f(value, Load<false>(params + i), _0));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            float32x4_t _slope = vdupq_n_f32(params[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, SynetRelu32f(value, _slope, _0));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetRelu32f(dst[j] + bias[i], params[i]);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetPreluLayerForward(dst, params, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationElu)
            {
                float alpha = params[0];
                if (bias)
                {
                    float32x4_t _alpha = vdupq_n_f32(alpha);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Neon::Elu(value, _alpha));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Neon::Elu(value, _alpha));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetElu32f(dst[j] + bias[i], alpha);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetElu32f(dst, size*count, &alpha, dst);
            }
            else if (activation == ::SimdConvolutionActivationHswish)
            {
                float shift = params[0];
                float scale = params[1];
                if (bias)
                {
                    float32x4_t _shift = vdupq_n_f32(shift);
                    float32x4_t _scale = vdupq_n_f32(scale);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Neon::SynetHswish32f(value, _shift, _scale));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Neon::SynetHswish32f(value, _shift, _scale));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetHswish32f(dst[j] + bias[i], shift, scale);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetHswish32f(dst, size*count, &shift, &scale, dst);
            }
            else if (activation == ::SimdConvolutionActivationMish)
            {
                float threshold = params[0];
                if (bias)
                {
                    float32x4_t _threshold = vdupq_n_f32(threshold);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Neon::Mish<1>(value, _threshold));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Neon::Mish<1>(value, _threshold));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetMish32f(dst[j] + bias[i], threshold);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetMish32f(dst, size * count, &threshold, dst);
            }
            else if (activation == ::SimdConvolutionActivationHardSigmoid)
            {
                float scale = params[0];
                float shift = params[1];
                if (bias)
                {
                    float32x4_t _scale = vdupq_n_f32(scale);
                    float32x4_t _shift = vdupq_n_f32(shift);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Neon::SynetHardSigmoid32f(value, _scale, _shift));
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
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Neon::SynetHardSigmoid32f(value, _scale, _shift));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetHardSigmoid32f(dst[j] + bias[i], scale, shift);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetHardSigmoid32f(dst, size * count, &scale, &shift, dst);
            }
            else if (activation == ::SimdConvolutionActivationSwish)
            {
                float threshold = params[0];
                if (bias)
                {
                    float32x4_t _threshold = vdupq_n_f32(threshold);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + i), Load<false>(bias + i));
                                Store<false>(dst + i, Neon::Swish<1>(value, _threshold));
                            }
                            for (; i < count; ++i)
                                dst[i] = Base::SynetSwish32f(dst[i] + bias[i], threshold);
                            dst += count;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            float32x4_t _bias = vdupq_n_f32(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                float32x4_t value = vaddq_f32(Load<false>(dst + j), _bias);
                                Store<false>(dst + j, Neon::Swish<1>(value, _threshold));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetSwish32f(dst[j] + bias[i], threshold);
                            dst += size;
                        }
                    }
                }
                else
                    Neon::SynetSwish32f(dst, size * count, &threshold, dst);
            }
            else
                assert(0);
        }

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam32f & p)
            : Base::SynetConvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Neon::Gemm32fNN, "Neon", p.gemm, "Ext"));
            if (_param.trans && _param.group == 1)
            {
                if (NHWC_GEMM_RUNTIME)
                {
#if defined(SIMD_ARM64_ENABLE)
                    _gemmCb.Init(InitGemmCbFuncs(Neon::Gemm32fNNcbBufferSize, Neon::Gemm32fNNcbReorderB, Neon::Gemm32fNNcbRun, "Neon", GemmKernelF2, GemmKernelF4));
#else
                    _gemmCb.Init(InitGemmCbFuncs(Neon::Gemm32fNNcbBufferSize, Neon::Gemm32fNNcbReorderB, Neon::Gemm32fNNcbRun, "Neon", GemmKernelF2, GemmKernelF3));
#endif
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M*_merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Neon::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Neon::Gemm32fNNcbRun;
                _nhwcReorderB = Neon::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam32f & p)
            : Base::SynetConvolution32fGemmNT(p)
        {
            _gemm.Init(InitGemmFuncs(Neon::Gemm32fNT, "Neon"));
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        bool SynetConvolution32fGemmNT::Preferable(const ConvParam32f & p)
        {
            if (p.group != 1)
                return false;
            if (p.trans)
                return p.Is1x1() && p.dstC == 1;
            else
                return p.srcH < 4 && p.srcW < 4;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam32f & p)
            : Base::SynetConvolution32fWinograd(p)
        {
            if (p.kernelY == 1 && p.kernelX == 3)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Neon::WinogradKernel1x3Block1x4SetFilter;
                    _setInput = Neon::WinogradKernel1x3Block1x4SetInput;
                    _setOutput = Neon::WinogradKernel1x3Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 1 && p.kernelX == 5)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Neon::WinogradKernel1x5Block1x4SetFilter;
                    _setInput = Neon::WinogradKernel1x5Block1x4SetInput;
                    _setOutput = Neon::WinogradKernel1x5Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 2 && p.kernelX == 2)
            {
                if (p.trans && p.srcH >= 8 && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 144)
                {
                    SetBlock(4, 4);
                    _setFilter = Neon::WinogradKernel2x2Block4x4SetFilter;
                    _setInput = Neon::WinogradKernel2x2Block4x4SetInput;
                    _setOutput = Neon::WinogradKernel2x2Block4x4SetOutput;
                }
                else
                {
                    SetBlock(2, 2);
                    _setFilter = Neon::WinogradKernel2x2Block2x2SetFilter;
                    _setInput = Neon::WinogradKernel2x2Block2x2SetInput;
                    _setOutput = Neon::WinogradKernel2x2Block2x2SetOutput;
                }
            }
            else if (p.kernelY == 3 && p.kernelX == 3)
            {
                if (p.trans && p.srcH >= 8 && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 144)
                {
                    SetBlock(4, 4);
                    _setFilter = Neon::WinogradKernel3x3Block4x4SetFilter;
                    _setInput = Neon::WinogradKernel3x3Block4x4SetInput;
                    _setOutput = Neon::WinogradKernel3x3Block4x4SetOutput;
                }
                else if (p.trans && p.srcH >= 6 && p.srcW >= 6 && p.srcH * p.srcW * p.batch >= 81 && p.dstH % 3 == 0 && p.dstW % 3 == 0)
                {
                    SetBlock(3, 3);
                    _setFilter = Neon::WinogradKernel3x3Block3x3SetFilter;
                    _setInput = Neon::WinogradKernel3x3Block3x3SetInput;
                    _setOutput = Neon::WinogradKernel3x3Block3x3SetOutput;
                }
                else
                {
                    SetBlock(2, 2);
                    _setFilter = Neon::WinogradKernel3x3Block2x2SetFilter;
                    _setInput = Neon::WinogradKernel3x3Block2x2SetInput;
                    _setOutput = Neon::WinogradKernel3x3Block2x2SetOutput;
                }
            }
            else
                assert(0);
            _gemm.Init(InitGemmFuncs(Neon::Gemm32fNN, "Neon", p.gemm, "Ext"));
            if (_param.trans)
            {
                if (NHWC_GEMM_RUNTIME)
                {
#if defined(SIMD_ARM64_ENABLE)
                    _gemmCb.Init(InitGemmCbFuncs(Neon::Gemm32fNNcbBufferSize, Neon::Gemm32fNNcbReorderB, Neon::Gemm32fNNcbRun, "Neon", GemmKernelF2, GemmKernelF4));
#else
                    _gemmCb.Init(InitGemmCbFuncs(Neon::Gemm32fNNcbBufferSize, Neon::Gemm32fNNcbReorderB, Neon::Gemm32fNNcbRun, "Neon", GemmKernelF2, GemmKernelF3));
#endif
                    _nhwcStrideW = _gemmCb.At(0).BufferSize(_M*_merge, _N, _K);
                }
                else
                    _nhwcStrideW = Neon::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                _nhwcWeight.Resize(_nhwcStrideW*_count);
                _nhwcRun = Neon::Gemm32fNNcbRun;
                _nhwcReorderB = Neon::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Neon::ConvolutionBiasAndActivation;
        }

        bool SynetConvolution32fWinograd::Preferable(const ConvParam32f & p)
        {
            if (!p.IsDilation(1) || !p.IsStride(1) || p.group != 1 || p.srcC < 10)
                return false;
            if (p.IsKernel(1, 3))
            {
                if (!(p.IsPad(0) || (p.padX == 1 && p.padW == 1)))
                    return false;
                if (p.srcC <= 32)
                    return false;
                return p.trans && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 36;
            }
            else if (p.IsKernel(1, 5))
            {
                if (!(p.IsPad(0) || (p.padX == 2 && p.padW == 2)))
                    return false;
                return p.trans && p.srcW >= 8 && p.srcH * p.srcW * p.batch >= 36;
            }
            else if (p.IsKernel(2))
            {
                if (!(p.IsPad(0) || (p.padY + p.padH == 1 && p.padX + p.padW == 1)))
                    return false;
                return p.trans && p.srcH >= 4 && p.srcW >= 4 && p.srcH * p.srcW * p.batch >= 36;
            }
            else if (p.IsKernel(3))
            {
                if (!(p.IsPad(0) || p.IsPad(1)))
                    return false;
                if (p.trans)
                    return p.srcH >= 4 && p.srcW >= 4 && p.srcH * p.srcW * p.batch >= 36;
                else
                    return p.srcH >= 6 && p.srcW >= 6;
            }
            return false;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fDepthwiseDotProduct::SynetConvolution32fDepthwiseDotProduct(const ConvParam32f & p)
            : Base::SynetConvolution32fDepthwiseDotProduct(p)
        {
        }

        SIMD_INLINE void DotProduct(const float * a, const float * b, size_t offset, float32x4_t & sum)
        {
            float32x4_t _a = Load<false>(a + offset);
            float32x4_t _b = Load<false>(b + offset);
            sum = vmlaq_f32(sum, _a, _b);
        }

        SIMD_INLINE float DotProduct(const float * a, const float * b, size_t size)
        {
            float sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                float32x4_t sums[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        DotProduct(a, b, i + F * 0, sums[0]);
                        DotProduct(a, b, i + F * 1, sums[1]);
                        DotProduct(a, b, i + F * 2, sums[2]);
                        DotProduct(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
                }
                for (; i < partialAlignedSize; i += F)
                    DotProduct(a, b, i, sums[0]);
                sum += ExtractSum32f(sums[0]);
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
                for (size_t n = 2; n <= 4; ++n)
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
            case 4 * F: return Set4r(p, a);
            default:
                return false;
            }
        }

        bool SynetConvolution32fNhwcDirect::Preferable(const ConvParam32f& p)
        {
            if (p.trans != SimdTrue || p.group != 1 || !p.IsDilation(1))
                return false;
            if (!p.Is1x1() && p.dstW < 6 + p.padX + p.padY)
                return false;
            if (p.Is1x1() && (p.srcC >= 2 * p.dstC || (p.activation == SimdConvolutionActivationIdentity && p.srcC > 128) || p.srcC > 256))
                return false;
            if (p.kernelY > p.srcH || p.kernelX > p.srcW)
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
            else if (SynetConvolution32fGemmNT::Preferable(param))
                return new SynetConvolution32fGemmNT(param);
            else if (SynetConvolution32fNhwcDirect::Preferable(param))
                return new SynetConvolution32fNhwcDirect(param);
            else if (SynetConvolution32fDirectNhwc::Preferable(param))
                return new SynetConvolution32fDirectNhwc(param);
            else
                return new SynetConvolution32fGemmNN(param);
        }
    }
#endif// SIMD_NEON_ENABLE
}

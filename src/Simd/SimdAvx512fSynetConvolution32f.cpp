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
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE)   
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
            else if (activation == ::SimdConvolutionActivationHardSigmoid)
            {
                float scale = params[0];
                float shift = params[1];
                if (bias)
                {
                    __m512 _scale = _mm512_set1_ps(scale);
                    __m512 _shift = _mm512_set1_ps(shift);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(dst + i);
                                __m512 _bias = _mm512_loadu_ps(bias + i);
                                _mm512_storeu_ps(dst + i, Avx512f::SynetHardSigmoid32f(_mm512_add_ps(_dst, _bias), _scale, _shift));
                            }
                            if (i < count)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + i);
                                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + i);
                                _mm512_mask_storeu_ps(dst + i, tail, Avx512f::SynetHardSigmoid32f(_mm512_add_ps(_dst, _bias), _scale, _shift));
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
                                _mm512_storeu_ps(dst + j, Avx512f::SynetHardSigmoid32f(value, _scale, _shift));
                            }
                            if (j < size)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _bias);
                                _mm512_mask_storeu_ps(dst + j, tail, Avx512f::SynetHardSigmoid32f(value, _scale, _shift));
                            }
                            dst += size;
                        }
                    }
                }
                else 
                    SynetHardSigmoid32f(dst, size * count, &scale, &shift, dst);
            }
            else if (activation == ::SimdConvolutionActivationSwish)
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
                                _mm512_storeu_ps(dst + i, Avx512f::Swish(_mm512_add_ps(_dst, _bias), _slope));
                            }
                            if (i < count)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + i);
                                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + i);
                                _mm512_mask_storeu_ps(dst + i, tail, Avx512f::Swish(_mm512_add_ps(_dst, _bias), _slope));
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
                                _mm512_storeu_ps(dst + j, Avx512f::Swish(value, _slope));
                            }
                            if (j < size)
                            {
                                __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _bias);
                                _mm512_mask_storeu_ps(dst + j, tail, Avx512f::Swish(value, _slope));
                            }
                            dst += size;
                        }
                    }
                }
                else
                    SynetSwish32f(dst, size * count, &slope, dst);
            }
            else
                assert(0);
#endif
        }

        //---------------------------------------------------------------------



        //---------------------------------------------------------------------

        SynetConvolution32fNhwcDirect::SynetConvolution32fNhwcDirect(const ConvParam32f& p)
            : Avx2::SynetConvolution32fNhwcDirect(p)
        {
            if (p.dstC <= Avx::F)
                return;
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
    }
#endif//SIMD_AVX512F_ENABLE
}

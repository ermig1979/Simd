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
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdGemm.h"

namespace Simd
{
#if defined(SIMD_AVX_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx
    {
        void ConvolutionBiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType activation, const float * params, ::SimdBool trans, float * dst)
        {
            size_t aligned = trans ? AlignLo(count, F) : AlignLo(size, F);
            if (activation == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    SynetAddBias(bias, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    __m256 _0 = _mm256_set1_ps(0.0f);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 _dst = _mm256_loadu_ps(dst + i);
                                __m256 _bias = _mm256_loadu_ps(bias + i);
                                _mm256_storeu_ps(dst + i, _mm256_max_ps(_0, _mm256_add_ps(_dst, _bias)));
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
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 _dst = _mm256_loadu_ps(dst + j);
                                _mm256_storeu_ps(dst + j, _mm256_max_ps(_0, _mm256_add_ps(_dst, _bias)));
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
                    __m256 _slope = _mm256_set1_ps(slope);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(bias + i));
                                _mm256_storeu_ps(dst + i, SynetRelu32f(value, _slope));
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
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + j), _bias);
                                _mm256_storeu_ps(dst + j, SynetRelu32f(value, _slope));
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
                    __m256 _lower = _mm256_set1_ps(lower);
                    __m256 _upper = _mm256_set1_ps(upper);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(bias + i));
                                _mm256_storeu_ps(dst + i, _mm256_min_ps(_mm256_max_ps(_lower, value), _upper));
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
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + j), _bias);
                                _mm256_storeu_ps(dst + j, _mm256_min_ps(_mm256_max_ps(_lower, value), _upper));
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
                        if (count == 1 || count == 2 || count == 4 || count == 8)
                        {
                            __m256 _bias, _slope;
                            if (count == 1)
                            {
                                _bias = _mm256_set1_ps(bias[0]);
                                _slope = _mm256_set1_ps(params[0]);
                            }
                            else if (count == 2)
                            {
                                _bias = _mm256_setr_ps(bias[0], bias[1], bias[0], bias[1], bias[0], bias[1], bias[0], bias[1]);
                                _slope = _mm256_setr_ps(params[0], params[1], params[0], params[1], params[0], params[1], params[0], params[1]);
                            }
                            else if (count == 4)
                            {
                                _bias = _mm256_setr_ps(bias[0], bias[1], bias[2], bias[3], bias[0], bias[1], bias[2], bias[3]);
                                _slope = _mm256_setr_ps(params[0], params[1], params[2], params[3], params[0], params[1], params[2], params[3]);
                            }
                            else if (count == 8)
                            {
                                _bias = _mm256_setr_ps(bias[0], bias[1], bias[2], bias[3], bias[4], bias[5], bias[6], bias[7]);
                                _slope = _mm256_setr_ps(params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]);
                            }
                            else
                                assert(0);
                            size_t n = size * count, nF = AlignLo(n, F), i = 0;
                            for (; i < nF; i += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + i), _bias);
                                _mm256_storeu_ps(dst + i, SynetRelu32f(value, _slope));
                            }
                            dst += nF;
                            for (size_t j = nF/count; j < size; ++j)
                            {
                                for (size_t i = 0; i < count; ++i)
                                    dst[i] = Base::SynetRelu32f(dst[i] + bias[i], params[i]);
                                dst += count;
                            }
                        }
                        else
                        {
                            for (size_t j = 0; j < size; ++j)
                            {
                                size_t i = 0;
                                for (; i < aligned; i += F)
                                {
                                    __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + i), _mm256_loadu_ps(bias + i));
                                    _mm256_storeu_ps(dst + i, SynetRelu32f(value, _mm256_loadu_ps(params + i)));
                                }
                                for (; i < count; ++i)
                                    dst[i] = Base::SynetRelu32f(dst[i] + bias[i], params[i]);
                                dst += count;
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < count; ++i)
                        {
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            __m256 _slope = _mm256_set1_ps(params[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 value = _mm256_add_ps(_mm256_loadu_ps(dst + j), _bias);
                                _mm256_storeu_ps(dst + j, SynetRelu32f(value, _slope));
                            }
                            for (; j < size; ++j)
                                dst[j] = Base::SynetRelu32f(dst[j] + bias[i], params[i]);
                            dst += size;
                        }
                    }
                }
                else
                    Avx::SynetPreluLayerForward(dst, params, count, size, dst, (SimdTensorFormatType)trans);
            }
            else if (activation == ::SimdConvolutionActivationHswish)
            {
                float shift = params[0];
                float scale = params[1];
                if (bias)
                {
                    __m256 _shift = _mm256_set1_ps(shift);
                    __m256 _scale = _mm256_set1_ps(scale);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 value = _mm256_add_ps(Load<false>(dst + i), Load<false>(bias + i));
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
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 value = _mm256_add_ps(Load<false>(dst + j), _bias);
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
            else if (activation == ::SimdConvolutionActivationHardSigmoid)
            {
                float scale = params[0];
                float shift = params[1];
                if (bias)
                {
                    __m256 _scale = _mm256_set1_ps(scale);
                    __m256 _shift = _mm256_set1_ps(shift);
                    if (trans)
                    {
                        for (size_t j = 0; j < size; ++j)
                        {
                            size_t i = 0;
                            for (; i < aligned; i += F)
                            {
                                __m256 value = _mm256_add_ps(Load<false>(dst + i), Load<false>(bias + i));
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
                            __m256 _bias = _mm256_set1_ps(bias[i]);
                            size_t j = 0;
                            for (; j < aligned; j += F)
                            {
                                __m256 value = _mm256_add_ps(Load<false>(dst + j), _bias);
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
            else
            {
                Sse2::ConvolutionBiasAndActivation(bias, count, size, activation, params, trans, dst);
            }
        }

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNN::SynetConvolution32fGemmNN(const ConvParam32f & p)
            : Sse2::SynetConvolution32fGemmNN(p)
        {
            _gemm.Init(InitGemmFuncs(Avx::Gemm32fNN, "Avx", p.gemm, "Ext"));
            if (_param.trans && _param.group == 1)
            {
                if (GemmRuntime())
                {
                    _gemmCb.Init(InitGemmCbFuncs(Avx::Gemm32fNNcbBufferSize, Avx::Gemm32fNNcbReorderB, Avx::Gemm32fNNcbRun, "Avx", GemmKernelF2, GemmKernelF3));
                    _nhwcWeight.Resize(_gemmCb.At(0).BufferSize(_M*_merge, _N, _K));
                }
                else
                    _nhwcWeight.Resize(Avx::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE));
                _nhwcRun = Avx::Gemm32fNNcbRun;
                _nhwcReorderB = Avx::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Avx::ConvolutionBiasAndActivation;
        }

        template<size_t size> SIMD_INLINE void Copy(const float * src, float * dst)
        {
            memcpy(dst, src, size * sizeof(float));
        }

        template<size_t size> SIMD_INLINE void Zero(float * dst)
        {
            memset(dst, 0, size * sizeof(float));
        }

        template<> SIMD_INLINE void Copy<16>(const float * src, float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_loadu_ps(src + 0 * F));
            _mm256_stream_ps(dst + 1 * F, _mm256_loadu_ps(src + 1 * F));
        }

        template<> SIMD_INLINE void Zero<16>(float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 1 * F, _mm256_setzero_ps());
        }

        template<> SIMD_INLINE void Copy<24>(const float * src, float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_loadu_ps(src + 0 * F));
            _mm256_stream_ps(dst + 1 * F, _mm256_loadu_ps(src + 1 * F));
            _mm256_stream_ps(dst + 2 * F, _mm256_loadu_ps(src + 2 * F));
        }

        template<> SIMD_INLINE void Zero<24>(float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 1 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 2 * F, _mm256_setzero_ps());
        }

        template<> SIMD_INLINE void Copy<32>(const float * src, float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_loadu_ps(src + 0 * F));
            _mm256_stream_ps(dst + 1 * F, _mm256_loadu_ps(src + 1 * F));
            _mm256_stream_ps(dst + 2 * F, _mm256_loadu_ps(src + 2 * F));
            _mm256_stream_ps(dst + 3 * F, _mm256_loadu_ps(src + 3 * F));
        }

        template<> SIMD_INLINE void Zero<32>(float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 1 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 2 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 3 * F, _mm256_setzero_ps());
        }

        template<> SIMD_INLINE void Copy<48>(const float * src, float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_loadu_ps(src + 0 * F));
            _mm256_stream_ps(dst + 1 * F, _mm256_loadu_ps(src + 1 * F));
            _mm256_stream_ps(dst + 2 * F, _mm256_loadu_ps(src + 2 * F));
            _mm256_stream_ps(dst + 3 * F, _mm256_loadu_ps(src + 3 * F));
            _mm256_stream_ps(dst + 4 * F, _mm256_loadu_ps(src + 4 * F));
            _mm256_stream_ps(dst + 5 * F, _mm256_loadu_ps(src + 5 * F));
        }

        template<> SIMD_INLINE void Zero<48>(float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 1 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 2 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 3 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 4 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 5 * F, _mm256_setzero_ps());
        }

        template<> SIMD_INLINE void Copy<64>(const float * src, float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_loadu_ps(src + 0 * F));
            _mm256_stream_ps(dst + 1 * F, _mm256_loadu_ps(src + 1 * F));
            _mm256_stream_ps(dst + 2 * F, _mm256_loadu_ps(src + 2 * F));
            _mm256_stream_ps(dst + 3 * F, _mm256_loadu_ps(src + 3 * F));
            _mm256_stream_ps(dst + 4 * F, _mm256_loadu_ps(src + 4 * F));
            _mm256_stream_ps(dst + 5 * F, _mm256_loadu_ps(src + 5 * F));
            _mm256_stream_ps(dst + 6 * F, _mm256_loadu_ps(src + 6 * F));
            _mm256_stream_ps(dst + 7 * F, _mm256_loadu_ps(src + 7 * F));
        }

        template<> SIMD_INLINE void Zero<64>(float * dst)
        {
            _mm256_stream_ps(dst + 0 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 1 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 2 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 3 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 4 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 5 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 6 * F, _mm256_setzero_ps());
            _mm256_stream_ps(dst + 7 * F, _mm256_setzero_ps());
        }

        template<size_t size> void ImgToCol(const ConvParam32f & p, const float * src, float * dst)
        {
            for (size_t g = 0; g < p.group; ++g)
            {
                for (size_t dy = 0; dy < p.dstH; ++dy)
                {
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                    {
                        for (size_t ky = 0; ky < p.kernelY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        Copy<size>(src + (sy * p.srcW + sx)*p.srcC, dst);
                                        dst += size;
                                    }
                                    else
                                    {
                                        Zero<size>(dst);
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                    Zero<size>(dst), dst += size;
                            }
                        }
                    }
                }
                src += size;
            }
        }

        void SynetConvolution32fGemmNN::ImgToRow(const float * src, float * dst)
        {
            const ConvParam32f & p = _param;
            assert(p.trans);
            size_t size = p.srcC / p.group;
            if (size*p.dstH*p.dstW*p.kernelY*p.kernelX >= 1024 * 512 && Aligned(dst))
            {
                if (size == 16)
                {
                    Avx::ImgToCol<16>(p, src, dst);
                    return;
                }
                if (size == 24)
                {
                    Avx::ImgToCol<24>(p, src, dst);
                    return;
                }
                if (size == 32)
                {
                    Avx::ImgToCol<32>(p, src, dst);
                    return;
                }
                if (size == 48)
                {
                    Avx::ImgToCol<48>(p, src, dst);
                    return;
                }
                if (size == 64)
                {
                    Avx::ImgToCol<64>(p, src, dst);
                    return;
                }
            }
            for (size_t g = 0; g < p.group; ++g)
            {
                for (size_t dy = 0; dy < p.dstH; ++dy)
                {
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                    {
                        for (size_t ky = 0; ky < p.kernelY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        memcpy(dst, src + (sy * p.srcW + sx)*p.srcC, size * sizeof(float));
                                        dst += size;
                                    }
                                    else
                                    {
                                        memset(dst, 0, size * sizeof(float));
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                memset(dst, 0, p.kernelX * size * sizeof(float));
                                dst += p.kernelX * size;
                            }
                        }
                    }
                }
                src += size;
            }
        }

        //---------------------------------------------------------------------

        SynetConvolution32fGemmNT::SynetConvolution32fGemmNT(const ConvParam32f & p)
            : Sse41::SynetConvolution32fGemmNT(p)
        {
            _gemm.Init(InitGemmFuncs(Avx::Gemm32fNT, "Avx"));
            _biasAndActivation = Avx::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fWinograd::SynetConvolution32fWinograd(const ConvParam32f & p)
            : Sse2::SynetConvolution32fWinograd(p)
        {
            if (p.kernelY == 1 && p.kernelX == 3)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Avx::WinogradKernel1x3Block1x4SetFilter;
                    _setInput = Avx::WinogradKernel1x3Block1x4SetInput;
                    _setOutput = Avx::WinogradKernel1x3Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 1 && p.kernelX == 5)
            {
                {
                    SetBlock(1, 4);
                    _setFilter = Avx::WinogradKernel1x5Block1x4SetFilter;
                    _setInput = Avx::WinogradKernel1x5Block1x4SetInput;
                    _setOutput = Avx::WinogradKernel1x5Block1x4SetOutput;
                }
            }
            else if (p.kernelY == 2 && p.kernelX == 2)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    SetBlock(4, 4);
                    _setFilter = Avx::WinogradKernel2x2Block4x4SetFilter;
                    _setInput = Avx::WinogradKernel2x2Block4x4SetInput;
                    _setOutput = Avx::WinogradKernel2x2Block4x4SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    SetBlock(2, 2);
                    _setFilter = Avx::WinogradKernel2x2Block2x2SetFilter;
                    _setInput = Avx::WinogradKernel2x2Block2x2SetInput;
                    _setOutput = Avx::WinogradKernel2x2Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else if (p.kernelY == 3 && p.kernelX == 3)
            {
                if (_blockY == 4 && _blockX == 4)
                {
                    _setFilter = Avx::WinogradKernel3x3Block4x4SetFilter;
                    _setInput = Avx::WinogradKernel3x3Block4x4SetInput;
                    _setOutput = Avx::WinogradKernel3x3Block4x4SetOutput;
                }
                else if (_blockY == 3 && _blockX == 3)
                {
                    _setFilter = Avx::WinogradKernel3x3Block3x3SetFilter;
                    _setInput = Avx::WinogradKernel3x3Block3x3SetInput;
                    _setOutput = Avx::WinogradKernel3x3Block3x3SetOutput;
                }
                else if (_blockY == 2 && _blockX == 2)
                {
                    _setFilter = Avx::WinogradKernel3x3Block2x2SetFilter;
                    _setInput = Avx::WinogradKernel3x3Block2x2SetInput;
                    _setOutput = Avx::WinogradKernel3x3Block2x2SetOutput;
                }
                else
                    assert(0);
            }
            else
                assert(0);
            _gemm.Init(InitGemmFuncs(Avx::Gemm32fNN, "Avx", p.gemm, "Ext"));
            if (_param.trans)
            {
                if (NHWC_GEMM_RUNTIME)
                {
                    _gemmCb.Init(InitGemmCbFuncs(Avx::Gemm32fNNcbBufferSize, Avx::Gemm32fNNcbReorderB, Avx::Gemm32fNNcbRun, "Avx", GemmKernelF2, GemmKernelF3));
                    _nhwcStrideW = _gemmCb.At(0).BufferSize(_M*_merge, _N, _K);
                }
                else
                    _nhwcStrideW = Avx::Gemm32fNNcbBufferSize(_M*_merge, _N, _K, GemmKernelAny, NHWC_GEMM_COMPATIBLE);
                _nhwcWeight.Resize(_nhwcStrideW*_count);
                _nhwcRun = Avx::Gemm32fNNcbRun;
                _nhwcReorderB = Avx::Gemm32fNNcbReorderB;
            }
            _biasAndActivation = Avx::ConvolutionBiasAndActivation;
        }

        //---------------------------------------------------------------------

        SynetConvolution32fDepthwiseDotProduct::SynetConvolution32fDepthwiseDotProduct(const ConvParam32f & p)
            : Sse2::SynetConvolution32fDepthwiseDotProduct(p)
        {
        }

        SIMD_INLINE void DotProduct(const float * a, const float * b, size_t offset, __m256 & sum)
        {
            __m256 _a = _mm256_loadu_ps(a + offset);
            __m256 _b = _mm256_loadu_ps(b + offset);
            sum = _mm256_add_ps(_mm256_mul_ps(_a, _b), sum);
        }

        SIMD_INLINE float DotProduct(const float * a, const float * b, size_t size)
        {
            float sum = 0;
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            if (partialAlignedSize)
            {
                __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                if (fullAlignedSize)
                {
                    for (; i < fullAlignedSize; i += QF)
                    {
                        DotProduct(a, b, i + F * 0, sums[0]);
                        DotProduct(a, b, i + F * 1, sums[1]);
                        DotProduct(a, b, i + F * 2, sums[2]);
                        DotProduct(a, b, i + F * 3, sums[3]);
                    }
                    sums[0] = _mm256_add_ps(_mm256_add_ps(sums[0], sums[1]), _mm256_add_ps(sums[2], sums[3]));
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
            : Sse2::SynetConvolution32fNhwcDirect(p)
        {
            if (p.dstC <= Sse2::F)
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

        //---------------------------------------------------------------------

        void * SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm)
        {
            if (conv->activation == SimdConvolutionActivationElu)
                return Sse2::SynetConvolution32fInit(batch, conv, gemm);
            ConvParam32f param(batch, conv, gemm);
            if (!param.Valid())
                return NULL;
            else if (SynetConvolution32fDepthwiseDotProduct::Preferable(param))
                return new SynetConvolution32fDepthwiseDotProduct(param);
            else if (SynetConvolution32fWinograd::Preferable(param))
                return new SynetConvolution32fWinograd(param);
            else if (SynetConvolution32fGemmNT::Preferable(param))
                return new SynetConvolution32fGemmNT(param);
            else if (SynetConvolution32fDirectNchw::Preferable(param))
                return new Avx::SynetConvolution32fDirectNchw(param);
            else if (SynetConvolution32fNhwcDirect::Preferable(param))
                return new SynetConvolution32fNhwcDirect(param);
            else if (SynetConvolution32fDirectNhwc::Preferable(param))
                return new SynetConvolution32fDirectNhwc(param);
            else
                return new SynetConvolution32fGemmNN(param);
        }
    }
#endif//SIMD_AVX_ENABLE
}

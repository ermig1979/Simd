/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)

    SynetConvolution16b::SynetConvolution16b(const ConvParam& p)
        : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        , _perf(NULL)
#endif
    {
        _src16b = p.srcT == SimdTensorData16b;
        _dst16b = p.dstT == SimdTensorData16b;
        _elemS = _src16b ? 2 : 4;
        _elemD = _dst16b ? 2 : 4;
        _is1x1 = p.Is1x1();
    }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
    Base::PerformanceMeasurer * SynetConvolution16b::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info(true) + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    void SynetConvolution16b::SetBias(const float* bias, size_t align)
    {
        const ConvParam& p = _param;
        _bias.Resize(AlignHi(p.dstC, align), true);
        if (bias)
            memcpy(_bias.data, bias, p.dstC * sizeof(float));
    }

    void SynetConvolution16b::SetParams(const float* params, size_t align)
    {
        const ConvParam& p = _param;
        if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
            _params.Resize(AlignHi(p.dstC, align), true);
        else
            _params.Resize(2, true);
        switch (p.activation)
        {
        case SimdConvolutionActivationIdentity:
            _params.data[0] = -FLT_MAX;
            _params.data[1] = FLT_MAX;
            break;
        case SimdConvolutionActivationRelu:
            _params.data[0] = 0;
            _params.data[1] = FLT_MAX;
            break;
        case SimdConvolutionActivationLeakyRelu:
            for (size_t d = 0; d < p.dstC; ++d)
                _params.data[d] = params[0];
            break;
        case SimdConvolutionActivationRestrictRange:
            _params.data[0] = params[0];
            _params.data[1] = params[1];
            break;
        case SimdConvolutionActivationPrelu:
            for (size_t d = 0; d < p.dstC; ++d)
                _params.data[d] = params[d];
            break;
        case SimdConvolutionActivationElu:
            _params.data[0] = params[0];
            break;
        case SimdConvolutionActivationHswish:
            _params.data[0] = params[0];
            _params.data[1] = params[1];
            break;
        case SimdConvolutionActivationMish:
            _params.data[0] = params[0];
            break;
        case SimdConvolutionActivationHardSigmoid:
            _params.data[0] = params[0];
            _params.data[1] = params[1];
            break;
        case SimdConvolutionActivationSwish:
            _params.data[0] = params[0];
            break;
        case SimdConvolutionActivationGelu:
            break;
        default:
            assert(0);
        }
    }

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        SynetConvolution16bGemm::SynetConvolution16bGemm(const ConvParam& p)
            : SynetConvolution16b(p)
        {
            if (p.trans)
            {
                _M = p.dstH * p.dstW;
                _N = p.dstC / p.group;
                _K = p.srcC * p.kernelY * p.kernelX / p.group;
                _ldS = _K;
                _ldW = p.dstC;
                _ldD = p.dstC;
                _grW = _N;
                _grS = _K * _M;
                _grD = _N;
                _weight.Resize(_K * _N * p.group);
            }
            else
            {
                _M = p.dstC / p.group;
                _N = p.dstH * p.dstW;
                _K = p.srcC * p.kernelY * p.kernelX / p.group;
                _ldW = _K;
                _ldS = _N;
                _ldD = _N;
                _grW = _M * _K;
                _grS = _K * _N;
                _grD = _M * _N;
                _weight.Resize(_K * _M * p.group);
            }
            _batch = p.batch;
            _sizeS = p.srcC * p.srcH * p.srcW;
            _sizeB = p.srcC * p.kernelY * p.kernelX * p.dstH * p.dstW;
            _sizeD = p.dstC * p.dstH * p.dstW;
            _stepS = _sizeS * _elemS;
            _stepD = _sizeD * _elemD;
        }

        size_t SynetConvolution16bGemm::ExternalBufferSize() const
        {
            size_t size = 0;
            if (!_src16b)
                size += _sizeS * sizeof(uint16_t);
            if (!_is1x1)
                size += _sizeB * sizeof(uint16_t);
            if (_dst16b)
                size += _sizeD * sizeof(float);
            return size;
        }

        void SynetConvolution16bGemm::SetParams(const float* weight, const float* bias, const float* params)
        {
            const ConvParam& p = _param;
            Float32ToBFloat16(weight, _weight.size, _weight.data);
            SynetConvolution16b::SetBias(bias, Alignment());
            SynetConvolution16b::SetParams(params, Alignment());
        }

        void SynetConvolution16bGemm::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const ConvParam& p = _param;
            buf = Buffer(buf);
            uint16_t* bufS = _src16b ? NULL : Allocate<uint16_t>(buf, _sizeS);
            uint16_t* bufB = _is1x1 ? NULL : Allocate<uint16_t>(buf, _sizeB);
            float* bufD = _dst16b ? Allocate<float>(buf, _sizeD) : NULL;
            const uint16_t* wgt = _weight.data;
            for (size_t b = 0; b < _batch; ++b)
            {
                const uint16_t* src16b = _src16b ? (uint16_t*)src : bufS;
                const uint16_t* buf16b = _is1x1 ? src16b : bufB;
                float* dst32f = _dst16b ? bufD : (float*)dst;
                if (!_src16b)
                    Float32ToBFloat16((float*)src, _sizeS, bufS);
                if (_param.trans)
                {
                    if(!_is1x1)
                        ImgToRow(src16b, bufB);
                    for (size_t g = 0; g < p.group; ++g)
                        GemmNN(_M, _N, _K, buf16b + _grS * g, _ldS, wgt + _grW * g, _ldW, dst32f + _grD * g, _ldD);
                }
                else
                {
                    if (!_is1x1)
                        ImgToCol(src16b, bufB);
                    for (size_t g = 0; g < p.group; ++g)
                        GemmNN(_M, _N, _K, wgt + _grW * g, _ldW, buf16b + _grS * g, _ldS, dst32f + _grD * g, _ldD);
                }
                ConvolutionBiasAndActivation(_bias.data, p.dstC, p.dstH * p.dstW, p.activation, _params.data, p.trans, dst32f);
                if(_dst16b)
                    Float32ToBFloat16(bufD, _sizeD, (uint16_t*)dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetConvolution16bGemm::ImgToCol(const uint16_t* src, uint16_t* dst)
        {
            const ConvParam& p = _param;
            assert(!p.trans);
            size_t srcSize = p.srcW * p.srcH;
            for (size_t c = 0; c < p.srcC; ++c)
            {
                for (size_t ky = 0; ky < p.kernelY; ky++)
                {
                    for (size_t kx = 0; kx < p.kernelX; kx++)
                    {
                        size_t sy = ky * p.dilationY - p.padY;
                        for (size_t dy = 0; dy < p.dstH; ++dy)
                        {
                            if (sy < p.srcH)
                            {
                                size_t sx = kx * p.dilationX - p.padX;
                                for (size_t dx = 0; dx < p.dstW; ++dx)
                                {
                                    if (sx < p.srcW)
                                        *(dst++) = src[sy * p.srcW + sx];
                                    else
                                        *(dst++) = 0;
                                    sx += p.strideX;
                                }
                            }
                            else
                            {
                                for (size_t dx = 0; dx < p.dstW; ++dx)
                                    *(dst++) = 0;
                            }
                            sy += p.strideY;
                        }
                    }
                }
                src += srcSize;
            }
        }

        void SynetConvolution16bGemm::ImgToRow(const uint16_t* src, uint16_t* dst)
        {
            const ConvParam& p = _param;
            assert(p.trans);
            size_t size = p.srcC / p.group;
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
                                        memcpy(dst, src + (sy * p.srcW + sx) * p.srcC, size * sizeof(uint16_t));
                                        dst += size;
                                    }
                                    else
                                    {
                                        memset(dst, 0, size * sizeof(uint16_t));
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                memset(dst, 0, p.kernelX * size * sizeof(uint16_t));
                                dst += p.kernelX * size;
                            }
                        }
                    }
                }
                src += size;
            }
        }

        void SynetConvolution16bGemm::GemmNN(size_t M, size_t N, size_t K, const uint16_t* A, size_t lda, const uint16_t* B, size_t ldb, float* C, size_t ldc)
        {
            for (size_t i = 0; i < M; ++i)
            {
                float* pC = C + i * ldc;
                for (size_t j = 0; j < N; ++j)
                    pC[j] = 0.0f;
                for (size_t k = 0; k < K; ++k)
                {
                    const uint16_t* pB = B + k * ldb;
                    float a = BFloat16ToFloat32(A[i * lda + k]);
                    for (size_t j = 0; j < N; ++j)
                        pC[j] += a * BFloat16ToFloat32(pB[j]);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void * SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam param(batch, conv, compatibility);
            if (!param.Valid(SimdTensorData32f, SimdTensorData16b))
                return NULL;
            if (Base::SynetConvolution16bNhwcDepthwise::Preferable(param))
                return new Base::SynetConvolution16bNhwcDepthwise(param);
            return new SynetConvolution16bGemm(param);
        }
    }
#endif
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynetDeconvolution16b.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    SynetDeconvolution16b::SynetDeconvolution16b(const DeconvParam& p)
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
    Base::PerformanceMeasurer* SynetDeconvolution16b::Perf(const char* func)
    {
        if (_perf == NULL)
            _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info(true) + " " + Desc(), Param().Flop());
        return _perf;
    }
#endif

    void SynetDeconvolution16b::SetBias(const float* bias, size_t align)
    {
        const DeconvParam& p = _param;
        _bias.Resize(AlignHi(p.dstC, align), true);
        if (bias)
            memcpy(_bias.data, bias, p.dstC * sizeof(float));
    }

    void SynetDeconvolution16b::SetParams(const float* params, size_t align)
    {
        const DeconvParam& p = _param;
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
        SynetDeconvolution16bGemm::SynetDeconvolution16bGemm(const DeconvParam& p)
            : SynetDeconvolution16b(p)
        {
            if (p.trans)
            {
                assert(p.group == 1);
                _M = p.srcH * p.srcW;
                _N = p.kernelY * p.kernelX * p.dstC;
                _K = p.srcC;
                _ldS = _K;
                _ldW = _N;
                _ldD = _N;
                _grW = 0;
                _grS = 0;
                _grD = 0;
                _weight.Resize(_K * _N);
            }
            else
            {
                _M = p.kernelY * p.kernelX * p.dstC / p.group;
                _N = p.srcH * p.srcW;
                _K = p.srcC / p.group;
                _ldW = _K;
                _ldS = _N;
                _ldD = _N;
                _grW = _M * _K;
                _grS = _K * _N;
                _grD = _M * _N;
                _weight.Resize(_K * _M);
            }
            _batch = p.batch;
            _sizeS = p.srcC * p.srcH * p.srcW;
            _sizeB = p.dstC * p.kernelY * p.kernelX * p.srcH * p.srcW;
            _sizeD = p.dstC * p.dstH * p.dstW;
            _stepS = _sizeS * _elemS;
            _stepD = _sizeD * _elemD;
        }

        size_t SynetDeconvolution16bGemm::ExternalBufferSize() const
        {
            size_t size = 0;
            if (!_src16b)
                size += _sizeS * sizeof(uint16_t);
            if (!_is1x1)
                size += _sizeB * sizeof(float);
            if (_dst16b)
                size += _sizeD * sizeof(float);
            return size;
        }

        void SynetDeconvolution16bGemm::SetParams(const float* weight, const float* bias, const float* params)
        {
            const DeconvParam& p = _param;
            if(p.trans)
                Float32ToBFloat16(weight, _weight.size, _weight.data);
            else
            {
                const float* src = weight;
                uint16_t* dst = _weight.data;
                for (size_t g = 0; g < _param.group; ++g)
                {
                    for (size_t i = 0; i < _M; ++i)
                        for (size_t k = 0; k < _K; ++k)
                            dst[i * _K + k] = Float32ToBFloat16(src[k * _M + i]);
                    src += _grW;
                    dst += _grW;
                }
            }
            SynetDeconvolution16b::SetBias(bias, Alignment());
            SynetDeconvolution16b::SetParams(params, Alignment());
        }

        void SynetDeconvolution16bGemm::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const DeconvParam& p = _param;
            buf = Buffer(buf);
            uint16_t* bufS = _src16b ? NULL : Allocate<uint16_t>(buf, _sizeS);
            float* bufB = _is1x1 ? NULL : Allocate<float>(buf, _sizeB);
            float* bufD = _dst16b ? Allocate<float>(buf, _sizeD) : NULL;
            const uint16_t* wgt = _weight.data;
            for (size_t b = 0; b < _batch; ++b)
            {
                const uint16_t* src16b = _src16b ? (uint16_t*)src : bufS;
                float* dst32f = _dst16b ? bufD : (float*)dst;
                float* buf32f = _is1x1 ? dst32f : bufB;
                if (!_src16b)
                    Float32ToBFloat16((float*)src, _sizeS, bufS);
                if (_param.trans)
                {
                    assert(p.group == 1);
                    GemmNN(_M, _N, _K, src16b, _ldS, wgt, _ldW, buf32f, _ldD);
                    if (!_is1x1)
                        ImgToRow(buf32f, dst32f);
                }
                else
                {
                    for (size_t g = 0; g < p.group; ++g)
                        GemmNN(_M, _N, _K, wgt + _grW * g, _ldW, src16b + _grS * g, _ldS, buf32f + _grD * g, _ldD);
                    if (!_is1x1)
                        ImgToCol(buf32f, dst32f);
                }
                ConvolutionBiasAndActivation(_bias.data, p.dstC, p.dstH * p.dstW, p.activation, _params.data, p.trans, dst32f);
                if (_dst16b)
                    Float32ToBFloat16(bufD, _sizeD, (uint16_t*)dst);
                src += _stepS;
                dst += _stepD;
            }
        }

        void SynetDeconvolution16bGemm::ImgToCol(const float* src, float* dst)
        {
            const DeconvParam& p = _param;
            assert(!p.trans);
            size_t dstSize = p.dstW * p.dstH;
            for (size_t cd = 0; cd < p.dstC; ++cd)
            {
                memset(dst, 0, dstSize * sizeof(float));
                for (size_t ky = 0; ky < p.kernelY; ++ky)
                {
                    for (size_t kx = 0; kx < p.kernelX; ++kx)
                    {
                        size_t dy = ky * p.dilationY - p.padY;
                        for (size_t sy = 0; sy < p.srcH; ++sy, dy += p.strideY)
                        {
                            if (dy < p.dstH)
                            {
                                size_t dx = kx * p.dilationX - p.padX;
                                for (size_t sx = 0; sx < p.srcW; ++sx, dx += p.strideX)
                                {
                                    if (dx < p.dstW)
                                        dst[dy * p.dstW + dx] += *src;
                                    src++;
                                }
                            }
                            else
                                src += p.srcW;
                        }
                    }
                }
                dst += dstSize;
            }
        }

        void SynetDeconvolution16bGemm::ImgToRow(const float* src, float* dst)
        {
            const DeconvParam& p = _param;
            assert(p.trans && p.group == 1);
            if (p.IsPad(0) && p.IsDilation(1) && p.kernelY == p.strideX && p.kernelX == p.strideX)
            {
                for (size_t sy = 0; sy < p.srcH; ++sy)
                {
                    for (size_t sx = 0; sx < p.srcW; ++sx)
                    {
                        size_t dy = sy * p.strideY;
                        for (size_t ky = 0; ky < p.kernelY; ky++, dy += 1)
                        {
                            size_t dx = sx * p.strideX;
                            for (size_t kx = 0; kx < p.kernelX; kx++, dx += 1)
                            {
                                memcpy(dst + (dy * p.dstW + dx) * p.dstC, src, p.dstC * sizeof(float));
                                src += p.dstC;
                            }
                        }
                    }
                }
            }
            else
            {
                for (size_t dy = 0; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        memset(dst + (dy * p.dstW + dx) * p.dstC, 0, p.dstC * sizeof(float));
                for (size_t sy = 0; sy < p.srcH; ++sy)
                {
                    for (size_t sx = 0; sx < p.srcW; ++sx)
                    {
                        size_t dy = sy * p.strideY - p.padY;
                        for (size_t ky = 0; ky < p.kernelY; ky++, dy += p.dilationY)
                        {
                            if (dy < p.dstH)
                            {
                                size_t dx = sx * p.strideX - p.padX;
                                for (size_t kx = 0; kx < p.kernelX; kx++, dx += p.dilationX)
                                {
                                    if (dx < p.dstW)
                                    {
                                        float* d = dst + (dy * p.dstW + dx) * p.dstC;
                                        for (size_t dc = 0; dc < p.dstC; ++dc)
                                            d[dc] += src[dc];
                                    }
                                    src += p.dstC;
                                }
                            }
                            else
                                src += p.kernelX * p.dstC;
                        }
                    }
                }
            }
        }

        void SynetDeconvolution16bGemm::GemmNN(size_t M, size_t N, size_t K, const uint16_t* A, size_t lda, const uint16_t* B, size_t ldb, float* C, size_t ldc)
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

        void* SynetDeconvolution16bInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            DeconvParam param(batch, conv, compatibility);
            if (!param.Valid(SimdTensorData16b))
                return NULL;
            return new SynetDeconvolution16bGemm(param);
        }
    }
#endif
}

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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        typedef SynetMergedConvolution16b::AlgParam AlgParam;
        typedef SynetMergedConvolution16b::ConvertToBf16Ptr ToBf16Ptr;
        typedef SynetMergedConvolution16b::InputConvolutionPtr InputPtr;
        typedef SynetMergedConvolution16b::DepthwiseConvolutionPtr DepthwisePtr;
        typedef SynetMergedConvolution16b::OutputConvolutionPtr OutputPtr;

        //-------------------------------------------------------------------------------------------------

        void ConvertFp32ToBf16(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t rowSize = p.srcW * p.srcC; 
            src += yBeg * rowSize;
            dst += yBeg * rowSize;
            for (size_t y = yBeg; y < yEnd; ++y)
            {
                for (size_t i = 0; i < rowSize; ++i)
                    dst[i] = Float32ToBFloat16(src[i]);
                src += rowSize;
                dst += rowSize;
            }
        }

        void ConvertBf16ToFp32(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, float* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t rowSize = p.srcW * p.srcC;
            for (size_t y = yBeg; y < yEnd; ++y)
                BFloat16ToFloat32(src + y * rowSize, rowSize, dst + y * rowSize);
        }

        void CopyFp32ToFp32(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, float* dst)
        {
            const float* src = (float*)src8;
            size_t rowSize = p.srcW * p.srcC;
            for (size_t y = yBeg; y < yEnd; ++y)
                memcpy(dst + y * rowSize, src + y * rowSize, rowSize * sizeof(float));
        }

        template<SimdConvolutionActivationType type> void InputConvolutionBf16Fp32(const uint16_t* src, const ConvParam& p, const AlgParam& a, 
            size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC, srcC2 = AlignLo(srcC, 2);
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            Array32f buf(dstC);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    buf.Clear();
                    for (size_t ky = 0; ky < kernelY; ++ky)
                    {
                        size_t sy = dy * strideY + ky - padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < kernelX; ++kx)
                            {
                                size_t sx = dx * strideX + kx - padX;
                                if (sx < p.srcW)
                                {
                                    const uint16_t* pw = weight + (ky * kernelX + kx) * srcC * dstC;
                                    const uint16_t* ps = src + (sy * srcW + sx) * srcC;
                                    for (size_t sc = 0; sc < srcC; ++sc)
                                    {
                                        float s = BFloat16ToFloat32(ps[sc]);
                                        for (size_t dc = 0; dc < dstC; ++dc)
                                            buf[dc] += s * BFloat16ToFloat32(pw[dc]);
                                        pw += dstC;
                                    }
                                }
                            }
                        }
                    }
                    for (size_t dc = 0; dc < dstC; ++dc)
                        dst[dc] = Activate<type>(buf[dc] + bias[dc], params, dc);
                    dst += p.dstC;
                }
            }
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolutionFp32Fp32(const uint8_t* src8, const ConvParam& p, const AlgParam& a, 
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst8)
        {
            assert(p.group == p.srcC && p.group == p.dstC);
            const float* src = (float*)src8;
            float* dst = (float*)dst8;
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float* pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const float* ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += ps[0] * pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Activate<type>(sum + bias[c], params, c);
                    }
                    dst += srcC;
                }
            }
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolutionFp32Bf16(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst8)
        {
            assert(p.group == p.srcC && p.group == p.dstC);
            const float* src = (float*)src8;
            uint16_t* dst = (uint16_t*)dst8;
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float* pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const float* ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += ps[0] * pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Float32ToBFloat16(Activate<type>(sum + bias[c], params, c));
                    }
                    dst += srcC;
                }
            }
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolutionBf16Bf16(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst8)
        {
            assert(p.group == p.srcC && p.group == p.dstC);
            const uint16_t* src = (uint16_t*)src8;
            uint16_t* dst = (uint16_t*)dst8;
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float* pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const uint16_t* ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += BFloat16ToFloat32(ps[0]) * pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Float32ToBFloat16(Activate<type>(sum + bias[c], params, c));
                    }
                    dst += srcC;
                }
            }
        }

		template<SimdConvolutionActivationType type> void OutputConvolutionBf16Fp32(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, int zero, const uint16_t* weight, const float* bias, const float* params, float* sum, uint8_t* dst8)
		{
            float* dst = (float*)dst8;
			size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
			Array32f buf(dstC);
			for (size_t dy = 0; dy < p.dstH; ++dy)
			{
				for (size_t dx = 0; dx < dstW; ++dx)
				{
                    if (zero)
                        buf.Clear();
                    else
                        buf.Assign(sum, dstC);
                    const uint16_t* pw = weight;
                    for (size_t sc = 0; sc < srcC; ++sc)
                    {
                        float s = BFloat16ToFloat32(src[sc]);
                        for (size_t dc = 0; dc < dstC; ++dc)
                            buf[dc] += s * BFloat16ToFloat32(pw[dc]);
                        pw += dstC;
                    }
                    for (size_t dc = 0; dc < dstC; ++dc)
                        dst[dc] = Activate<type>(buf[dc] + bias[dc], params, dc);
                    src += srcC;
                    sum += dstC;
                    dst += dstC;
				}
			}
		}

        template<SimdConvolutionActivationType type> void OutputConvolutionBf16Bf16(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, int zero, const uint16_t* weight, const float* bias, const float* params, float* sum, uint8_t* dst8)
        {
            uint16_t* dst = (uint16_t*)dst8;
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW, dstC = p.dstC;
            Array32f buf(dstC);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    if (zero)
                        buf.Clear();
                    else
                        buf.Assign(sum, dstC);
                    const uint16_t* pw = weight;
                    for (size_t sc = 0; sc < srcC; ++sc)
                    {
                        float s = BFloat16ToFloat32(src[sc]);
                        for (size_t dc = 0; dc < dstC; ++dc)
                            buf[dc] += s * BFloat16ToFloat32(pw[dc]);
                        pw += dstC;
                    }
                    for (size_t dc = 0; dc < dstC; ++dc)
                        dst[dc] = Float32ToBFloat16(Activate<type>(buf[dc] + bias[dc], params, dc));
                    src += srcC;
                    sum += dstC;
                    dst += dstC;
                }
            }
        }

        template <SimdConvolutionActivationType type> void Set(const MergConvParam& p, size_t index, InputPtr & input, DepthwisePtr & depthwise, OutputPtr & output)
        {
            switch (index)
            {
            case 0:
                if (p.conv[0].group == 1)
                    input = InputConvolutionBf16Fp32<type>;
                else
                    depthwise = p.conv[0].srcT == SimdTensorData16b ? DepthwiseConvolutionBf16Bf16<type> : DepthwiseConvolutionFp32Bf16<type>;
                break;
            case 1:
                if (p.conv[1].group == 1)
                    output = p.conv[1].dstT == SimdTensorData16b ? OutputConvolutionBf16Bf16<type> : OutputConvolutionBf16Fp32<type>;
                else
                    depthwise = p.conv[1].dstT == SimdTensorData16b ? DepthwiseConvolutionFp32Bf16<type> : DepthwiseConvolutionFp32Fp32<type>;
                break;
            case 2:
                output = p.conv[2].dstT == SimdTensorData16b ? OutputConvolutionBf16Bf16<type> : OutputConvolutionBf16Fp32<type>;
                break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SynetMergedConvolution16b::SynetMergedConvolution16b(const MergConvParam& p)
            : _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
            , _perf(NULL)
#endif
        {
            memset(&_alg, 0, sizeof(_alg));
            _toBf16 = NULL, _toFp32 = NULL, _input = NULL, _depthwise = NULL, _output[0] = NULL, _output[1] = NULL;
            const ConvParam& beg = p.conv[0];
            const ConvParam& end = p.conv[p.count - 1];
            _dw0 = beg.group != 1;
            const ConvParam& dw = _dw0 ? p.conv[0] : p.conv[1];
            _src16b = beg.srcT == SimdTensorData16b;
            _dst16b = end.dstT == SimdTensorData16b;
            _alg.elem[0] = _src16b ? 2 : 4;
            _alg.elem[1] = _dst16b ? 2 : 4;
            _sizeS = beg.srcH * beg.srcW * beg.srcC;
            _sizeD = end.dstH * end.dstW * end.dstC;

            _sizeB[0] = _dw0 || _src16b ? 0 : p.conv[0].srcH * p.conv[0].srcW * p.conv[0].srcC;
            _sizeB[1] = _dw0 ? 0 : dw.srcH * dw.srcW * dw.srcC;
            _sizeB[2] = _dw0 || p.count == 3 ? dw.dstH * dw.dstW * dw.dstC : 0;
            _sizeB[3] = _dst16b && (end.group == 1 || p.add) ? end.dstH * end.dstW * end.dstC : 0;
            _toBf16 = ConvertFp32ToBf16;
            _toFp32 = _src16b ? ConvertBf16ToFp32 : CopyFp32ToFp32;
            for (size_t i = 0; i < p.count; ++i)
            {
                switch (p.conv[i].activation)
                {
                case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationIdentity>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRelu>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationLeakyRelu>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(_param, i, _input, _depthwise, _output[0]); break;
                case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(_param, i, _input, _depthwise, _output[0]); break;
                default: assert(0);
                }
            }
        }

        size_t SynetMergedConvolution16b::ExternalBufferSize() const
        {
            return (_sizeB[0] + _sizeB[2]) * 2 + (_sizeB[1] + _sizeB[3]) * 4 + SIMD_ALIGN * 2;
        }

        size_t SynetMergedConvolution16b::InternalBufferSize() const
        {
            size_t size = _buffer.RawSize() + _weightD.RawSize() + _weightI.RawSize() + _weightO.RawSize();
            for (size_t i = 0; i < _param.count; ++i)
                size += _bias[i].RawSize() + _params[i].RawSize();
            return size;
        }

        void SynetMergedConvolution16b::SetParams(const float* const* weight, const float* const* bias, const float* const* params)
        {
            const MergConvParam& p = _param;
            if (_dw0)
            {
                SetDepthwiseWeight(weight[0], p.conv[0]);
                SetOutputWeight(weight[1], p.conv[1]);
            }
            else
            {
                SetInputWeight(weight[0], p.conv[0]);
                SetDepthwiseWeight(weight[1], p.conv[1]);
                if(p.count > 2)
                    SetOutputWeight(weight[2], p.conv[2]);
            }
            for (size_t i = 0; i < p.count; ++i)
            {
                SetBias(bias[i], p.conv[i], _bias[i]);
                SetParams(params[i], p.conv[i], _params[i]);
            }
        }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* SynetMergedConvolution16b::Perf(const char* func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info(true) + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        const char* SynetMergedConvolution16b::Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

        void SynetMergedConvolution16b::SetInputWeight(const float* src, const ConvParam& p)
        {
            assert(p.group == 1);
            if (_alg.miC)
            {
                assert(Is1x1(p));
                size_t F = _alg.miC, C = AlignHi(p.srcC, _alg.miK), D = DivHi(p.dstC, F);
                _weightI.Resize(C * D * F, true);
                uint16_t* dst = _weightI.data;
                for (size_t d = 0; d < D; d++)
                {
                    for (size_t c = 0; c < C; c += 2)
                    {
                        const float* ps = src + c * p.dstC + d * F;
                        for (size_t f = 0; f < F; ++f)
                        {
                            for (size_t i = 0; i < 2; ++i)
                            {
                                if (d * F + f < p.dstC && c + i < p.srcC)
                                    *(dst++) = Float32ToBFloat16(ps[i * p.dstC]);
                                else
                                    *(dst++) = 0;
                            }
                            if(c < p.srcC)
                                ps++;
                        }
                    }
                }            
            }
            else
            {
                _weightI.Resize(p.kernelY * p.kernelX * p.srcC * p.dstC, true);
                Float32ToBFloat16(src, _weightI.size, _weightI.data);
            }
        }

        void SynetMergedConvolution16b::SetDepthwiseWeight(const float* src, const ConvParam& p)
        {
            assert(p.srcC == p.dstC && p.srcC == p.group);
            if (_alg.miC)
            {
                size_t D = p.dstC, K = p.kernelY * p.kernelX, F = _alg.miC;
                _weightD.Resize(AlignHiAny(D, F) * K);
                float* dst = _weightD.data;
                for (size_t d = 0; d < D; d += F)
                {
                    size_t n = Simd::Min(F, D - d);
                    for (size_t k = 0; k < K; k++)
                    {
                        size_t i = 0;
                        for (; i < n; ++i)
                            dst[i] = src[k * D + d + i];
                        for (; i < F; ++i)
                            dst[i] = 0;
                        dst += F;
                    }
                }
            }
            else
                _weightD.Assign(src, p.kernelY * p.kernelX * p.srcC * p.dstC / p.group);
        }

        void SynetMergedConvolution16b::SetOutputWeight(const float* src, const ConvParam& p)
        {
            assert(p.group == 1 && Is1x1(p));
            if (_alg.miC)
            {
                size_t F = _alg.miC, C = DivHi(AlignHi(p.srcC, _alg.miK), 2), D = DivHi(p.dstC, F), M = DivHi(_alg.maC, 2);
                _weightO.Resize(C * D * F * 2, true);
                uint16_t* dst = _weightO.data;
                for (size_t cB = 0; cB < C; cB += M)
                {
                    size_t cE = Simd::Min(C, cB + M);
                    for (size_t d = 0; d < D; d++)
                    {
                        for (size_t c = cB; c < cE; ++c)
                        {
                            const float* ps = src + c * 2 * p.dstC + d * F;
                            for (size_t f = 0; f < F; ++f)
                            {
                                for (size_t i = 0; i < 2; ++i)
                                {
                                    if (d * F + f < p.dstC && c * 2 + i < p.srcC)
                                        *(dst++) = Float32ToBFloat16(ps[i * p.dstC]);
                                    else
                                        *(dst++) = 0;
                                }
                                if (c * 2 < p.srcC)
                                    ps++;
                            }
                        }
                    }
                }
            }
            else
            {
                _weightO.Resize(p.kernelY * p.kernelX * p.srcC * p.dstC, true);
                Float32ToBFloat16(src, _weightO.size, _weightO.data);
            }
        }

        void SynetMergedConvolution16b::SetBias(const float* src, const ConvParam& p, Array32f& dst)
        {
            const AlgParam& a = _alg;
            dst.Resize(AlignHiAny(p.dstC, Simd::Max(size_t(1), a.miC * 2)), true);
            if (src)
                memcpy(dst.data, src, p.dstC * sizeof(float));
        }

        void SynetMergedConvolution16b::SetParams(const float* src, const ConvParam& p, Array32f& dst)
        {
            const AlgParam& a = _alg;
            if (p.activation == SimdConvolutionActivationLeakyRelu || p.activation == SimdConvolutionActivationPrelu)
                dst.Resize(AlignHiAny(p.dstC, Simd::Max(size_t(1), a.miC * 2)), true);
            else
                dst.Resize(2, true);
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity:
                dst.data[0] = -FLT_MAX;
                dst.data[1] = FLT_MAX;
                break;
            case SimdConvolutionActivationRelu:
                dst.data[0] = 0;
                dst.data[1] = FLT_MAX;
                break;
            case SimdConvolutionActivationLeakyRelu:
                for (size_t d = 0; d < p.dstC; ++d)
                    dst.data[d] = src[0];
                break;
            case SimdConvolutionActivationRestrictRange:
                dst.data[0] = src[0];
                dst.data[1] = src[1];
                break;
            case SimdConvolutionActivationPrelu:
                for (size_t d = 0; d < p.dstC; ++d)
                    dst.data[d] = src[d];
                break;
            case SimdConvolutionActivationElu:
                dst.data[0] = src[0];
                break;
            case SimdConvolutionActivationHswish:
                dst.data[0] = src[0];
                dst.data[1] = src[1];
                break;
            case SimdConvolutionActivationMish:
                dst.data[0] = src[0];
                break;
            case SimdConvolutionActivationHardSigmoid:
                dst.data[0] = src[0];
                dst.data[1] = src[1];
                break;
            case SimdConvolutionActivationSwish:
                dst.data[0] = src[0];
                break;
            case SimdConvolutionActivationGelu:
                break;
            default:
                assert(0);
            }
        }

        void SynetMergedConvolution16b::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            buf = Buffer(buf);
            uint16_t* buf0 = Allocate<uint16_t>(buf, _sizeB[0]);
            float* buf1 = Allocate<float>(buf, _sizeB[1]);
            uint16_t* buf2 = Allocate<uint16_t>(buf, _sizeB[2]);
            float* buf3 = Allocate<float>(buf, _sizeB[3]);
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            const ConvParam& c2 = p.conv[2];
            const AlgParam& a = _alg;
            for (size_t b = 0; b < c0.batch; ++b)
            {
                if (_dw0)
                {
                    _depthwise(src, c0, a, 0, 0, c0.dstH, _weightD.data, _bias[0].data, _params[0].data, (uint8_t*)buf2);
                    _output[0](buf2, c1, a, 0, 0, c1.dstH, 1, _weightO.data, _bias[1].data, _params[1].data, buf3, dst);
                }
                else
                {
                    if (!_src16b)
                        _toBf16(src, c0, a, 0, c0.srcH, buf0);
                    const uint16_t* src16b = _src16b ? (uint16_t*)src : buf0;
                    _input(src16b, c0, a, 0, 0, c0.dstH, _weightI.data, _bias[0].data, _params[0].data, buf1);
                    if (p.count > 2)
                    {
                        _depthwise((uint8_t*)buf1, c1, a, 0, 0, c1.dstH, _weightD.data, _bias[1].data, _params[1].data, (uint8_t*)buf2);
                        if (!_dst16b)
                            buf3 = (float*)dst;
                        if(p.add)
                            _toFp32(src, c0, a, 0, c0.srcH, buf3);
                        _output[0](buf2, c2, a, 0, 0, c2.dstH, p.add ? 0 : 1, _weightO.data, _bias[2].data, _params[2].data, buf3, dst);
                    }
                    else
                        _depthwise((uint8_t*)buf1, c1, a, 0, 0, c1.dstH, _weightD.data, _bias[1].data, _params[1].data, dst);
                }
                src += _sizeS * _alg.elem[0];
                dst += _sizeD * _alg.elem[1];
            }
        };

        uint8_t* SynetMergedConvolution16b::Buffer(uint8_t* buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution16bCdc::SynetMergedConvolution16bCdc(const MergConvParam& p)
            : SynetMergedConvolution16b(p)
        {
        }

        void SynetMergedConvolution16bCdc::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            const ConvParam& c2 = p.conv[2];
            const AlgParam& a = _alg;

            buf = Buffer(buf);
            uint16_t* buf0 = Allocate<uint16_t>(buf, _sizeB[0]);
            SetGap(buf);
            float* buf1 = Allocate<float>(buf, _sizeB[1]);
            uint16_t* buf2 = Allocate<uint16_t>(buf, _sizeB[2]);
            SetGap(buf);
            float* buf3 = Allocate<float>(buf, _sizeB[3]);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c1.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c1.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c1.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c1.srcH);
                        size_t yEnd0 = Simd::RestrictRange(yBeg0 + a.yStep[0], a.yStart[0], c0.srcH);
                        if (_toBf16)
                            _toBf16(src, c0, a, yBeg0, yEnd0, buf0);
                        const uint16_t* src16b = _toBf16 ? buf0 : (uint16_t*)src;
                        _input(src16b, c0, a, maC, yBeg1, yEnd1, _weightI.data + c * a.dw[0],
                            _bias[0].data + c, _params[0].data + c * a.dp[0], buf1);
                        _depthwise((uint8_t*)buf1, c1, a, maC, yBeg2, yEnd2, _weightD.data + c * a.dw[1],
                            _bias[1].data + c, _params[1].data + c * a.dp[1], (uint8_t*)buf2);
                        float *buf3p = buf3 == NULL ? (float*)dst : buf3;
                        if (p.add && c == 0)
                            _toFp32(src, c0, a, yBeg2, yEnd2, buf3p);
                        if (c + maC == C)
                            _output[0](buf2, c2, a, maC, yBeg2, yEnd2, (maC != C || p.add) ? 0 : 1,
                                _weightO.data + c * a.dw[2], _bias[2].data, _params[2].data, buf3p, dst);
                        else
                            _output[1](buf2, c2, a, maC, yBeg2, yEnd2, (c != 0 || p.add) ? 0 : 1,
                                _weightO.data + c * a.dw[2], _bias[2].data, _params[2].data, buf3p, dst);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS * a.elem[0];
                dst += _sizeD * a.elem[1];
            }
        }

        bool SynetMergedConvolution16bCdc::Preferable(const MergConvParam& p)
        {
            return p.count == 3 && Is1x1(p.conv[0]);
        }

        void SynetMergedConvolution16bCdc::SetSize(size_t miC, size_t miK)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            const ConvParam& c2 = p.conv[2];
            AlgParam& a = _alg;

            a.miC = miC;
            a.miK = miK;
            size_t size = 0;
            for (size_t i = 0; i < 3; ++i)
            {
                const ConvParam& c = p.conv[i];
                if (c.group == 1)
                    size += AlignHi(c.srcC, a.miK) * AlignHi(c.dstC, a.miC * 2) * 2;
                else
                    size += c.kernelY * c.kernelX * AlignHi(c.srcC, a.miC) * 4;
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHi(AlignHi(c0.dstC / count, 2 * a.miC), a.miK);
            for (size_t yStep = c1.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];
                a.bufH[2] = Pow2Hi(a.yStep[2]);

                a.yStep[1] = a.yStep[2] * c1.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c1.strideY + c1.kernelY - c1.padY, c1.srcH);
                a.bufH[1] = Pow2Hi(Simd::Max((a.yStep[2] - 1) * c1.strideY + c1.kernelY, a.yStart[1]));

                a.yStep[0] = a.yStep[1];
                a.yStart[0] = Simd::Min(a.yStart[1], c0.srcH);
                a.bufH[0] = _src16b && Aligned(c0.srcC, a.miK) ? 0 : Simd::Max(a.yStep[0], a.yStart[0]);

                _sizeB[0] = _src16b && Aligned(c0.srcC, a.miK) ? 0 : a.bufH[0] * p.conv[0].srcW * AlignHi(p.conv[0].srcC, a.miK);
                _sizeB[1] = a.bufH[1] * p.conv[1].srcW * a.maC;
                _sizeB[2] = a.bufH[2] * p.conv[1].dstW * a.maC;
                if (_sizeB[0] * 2 + _sizeB[1] * 4 + _sizeB[2] * 2 <= L2)
                    break;
            }
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = AlignHi(c0.srcC, a.miK);
            a.dw[1] = c1.kernelY * c1.kernelX;
            a.dw[2] = AlignHi(c2.dstC, a.miC);
            _sizeB[3] = _dst16b && (count > 1 || p.add || a.miK == 32) ? _sizeD : 0;
            
            ((ConvParam&)c1).dstT = SimdTensorData16b;
            ((ConvParam&)c2).srcT = SimdTensorData16b;
        }

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution16bCd::SynetMergedConvolution16bCd(const MergConvParam& p)
            : SynetMergedConvolution16b(p)
        {
        }

        void SynetMergedConvolution16bCd::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            const AlgParam& a = _alg;

            buf = Buffer(buf);
            uint16_t* buf0 = Allocate<uint16_t>(buf, _sizeB[0]);
            SetGap(buf);
            float* buf1 = Allocate<float>(buf, _sizeB[1]);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c1.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c1.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c1.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c1.srcH);
                        size_t yEnd0 = Simd::RestrictRange(yBeg0 + a.yStep[0], a.yStart[0], c0.srcH);
                        if(_toBf16)
                            _toBf16(src, c0, a, yBeg0, yEnd0, buf0);
                        const uint16_t* src16b = _toBf16 ? buf0 : (uint16_t*)src;
                        _input(src16b, c0, a, maC, yBeg1, yEnd1, _weightI.data + c * a.dw[0],
                            _bias[0].data + c, _params[0].data + c * a.dp[0], buf1);
                        _depthwise((uint8_t*)buf1, c1, a, maC, yBeg2, yEnd2, _weightD.data + c * a.dw[1],
                            _bias[1].data + c, _params[1].data + c * a.dp[1], dst + c * a.elem[1]);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                        yBeg0 = yEnd0;
                    }
                }
                src += _sizeS * a.elem[0];
                dst += _sizeD * a.elem[1];
            }
        }

        bool SynetMergedConvolution16bCd::Preferable(const MergConvParam& p)
        {
            return p.count == 2 && p.conv[0].group == 1 && Is1x1(p.conv[0]);
        }

        void SynetMergedConvolution16bCd::SetSize(size_t miC, size_t miK)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            AlgParam& a = _alg;

            a.miC = miC;
            a.miK = miK;
            size_t size = 0;
            for (size_t i = 0; i < 2; ++i)
            {
                const ConvParam& c = p.conv[i];
                if (c.group == 1)
                    size += AlignHi(c.srcC, a.miK) * AlignHi(c.dstC, a.miC * 2) * 2;
                else
                    size += c.kernelY * c.kernelX * AlignHi(c.srcC, a.miC) * 4;
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHiAny(c0.dstC / count, 2 * a.miC);
            for (size_t yStep = c1.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];

                a.yStep[1] = a.yStep[2] * c1.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c1.strideY + c1.kernelY - c1.padY, c1.srcH);
                a.bufH[1] = Pow2Hi(Simd::Max((a.yStep[2] - 1) * c1.strideY + c1.kernelY, a.yStart[1]));

                a.yStep[0] = a.yStep[1];
                a.yStart[0] = Simd::Min(a.yStart[1], c0.srcH);
                a.bufH[0] = _src16b && Aligned(c0.srcC, a.miK) ? 0 : Simd::Max(a.yStep[1], a.yStart[0]);

                _sizeB[0] = a.bufH[0] * p.conv[0].srcW * AlignHi(p.conv[0].srcC, a.miK);
                _sizeB[1] = a.bufH[1] * p.conv[1].srcW * a.maC;
                if (_sizeB[0] * 2 + _sizeB[1] * 4 <= L2)
                    break;
            }
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = AlignHi(c0.srcC, a.miK);
            a.dw[1] = c1.kernelY * c1.kernelX;
            a.dw[2] = 0;
            a.bufH[2] = 0;
            _sizeB[2] = 0;
            _sizeB[3] = 0;
        }

        //-----------------------------------------------------------------------------------------

        SynetMergedConvolution16bDc::SynetMergedConvolution16bDc(const MergConvParam& p)
            : SynetMergedConvolution16b(p)
        {
        }

        void SynetMergedConvolution16bDc::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            const AlgParam& a = _alg;

            buf = Buffer(buf);
            uint16_t* buf2 = Allocate<uint16_t>(buf, _sizeB[2]);
            float* buf3 = Allocate<float>(buf, _sizeB[3]);
            SetGap(buf);

            for (size_t b = 0; b < c0.batch; ++b)
            {
                for (size_t c = 0, C = c0.dstC; c < C; c += a.maC)
                {
                    size_t maC = Simd::Min(C, c + a.maC) - c;
                    for (size_t yBeg2 = 0, yBeg1 = 0, yBeg0 = 0; yBeg2 < c1.dstH;)
                    {
                        size_t yEnd2 = Simd::RestrictRange(yBeg2 + a.yStep[2], a.yStart[2], c0.dstH);
                        size_t yEnd1 = Simd::RestrictRange(yBeg1 + a.yStep[1], a.yStart[1], c0.srcH);
                        _depthwise(src + c * a.elem[0], c0, a, maC, yBeg2, yEnd2, _weightD.data + c * a.dw[0], _bias[0].data + c,
                            _params[0].data + c * a.dp[0], (uint8_t*)buf2);
                        float* buf3p = buf3 == NULL ? (float*)dst : buf3;
                        if (c + maC == C)
                            _output[0](buf2, c1, a, maC, yBeg2, yEnd2, maC != C ? 0 : 1, _weightO.data + c * a.dw[1],
                                _bias[1].data, _params[1].data, buf3p, dst);
                        else
                            _output[1](buf2, c1, a, maC, yBeg2, yEnd2, c != 0 ? 0 : 1, _weightO.data + c * a.dw[1],
                                _bias[1].data, _params[1].data, buf3p, dst);
                        yBeg2 = yEnd2;
                        yBeg1 = yEnd1;
                    }
                }
                src += _sizeS * a.elem[0];
                dst += _sizeD * a.elem[1];
            }
        }

        bool SynetMergedConvolution16bDc::Preferable(const MergConvParam& p)
        {
            return p.count == 2 && p.conv[1].group == 1;
        }

        void SynetMergedConvolution16bDc::SetSize(size_t miC, size_t miK)
        {
            const size_t L1 = Base::AlgCacheL1(), L2 = Base::AlgCacheL2(), L3 = Base::AlgCacheL3();
            const MergConvParam& p = _param;
            const ConvParam& c0 = p.conv[0];
            const ConvParam& c1 = p.conv[1];
            AlgParam& a = _alg;

            a.miC = miC;
            a.miK = miK;
            size_t size = 0;
            for (size_t i = 0; i < 2; ++i)
            {
                const ConvParam& c = p.conv[i];
                if (c.group == 1)
                    size += AlignHi(c.srcC, a.miK) * AlignHi(c.dstC, a.miC * 2) * 2;
                else
                    size += c.kernelY * c.kernelX * AlignHi(c.srcC, a.miC) * 4;
            }
            size_t count = size / (L3 / 2) + 1;
            a.maC = AlignHi(AlignHi(c0.dstC / count, 2 * a.miC), a.miK);
            for (size_t yStep = c0.dstH; yStep >= 1; yStep--)
            {
                a.yStep[2] = Simd::Max<size_t>(1, yStep);
                a.yStart[2] = a.yStep[2];
                a.bufH[2] = Pow2Hi(a.yStep[2]);

                a.yStep[1] = a.yStep[2] * c0.strideY;
                a.yStart[1] = Simd::Min((a.yStart[2] - 1) * c0.strideY + c0.kernelY - c0.padY, c0.srcH);

                _sizeB[2] = a.bufH[2] * p.conv[1].srcW * a.maC;
                if (_sizeB[2] * 2 <= L2)
                    break;
            }
            a.bufH[0] = 0;
            a.bufH[1] = 0;
            _sizeB[0] = 0;
            _sizeB[1] = 0;
            _sizeB[3] = _dst16b && (count > 1 || a.miK == 32) ? _sizeD : 0;
            a.dp[0] = c0.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dp[1] = c1.activation == ::SimdConvolutionActivationPrelu ? 1 : 0;
            a.dw[0] = c0.kernelY * c0.kernelX;
            a.dw[1] = AlignHi(c1.dstC, a.miC);

            ((ConvParam&)c0).dstT = SimdTensorData16b;
            ((ConvParam&)c1).srcT = SimdTensorData16b;
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetMergedConvolution16bInit(size_t batch, const SimdConvolutionParameters* convs, size_t count, SimdBool add)
        {
            MergConvParam param(batch, convs, count, add);
            if (!param.Valid(SimdTensorData32f, SimdTensorData16b))
                return NULL;
            return new SynetMergedConvolution16b(param);
        }
    }
#endif
}

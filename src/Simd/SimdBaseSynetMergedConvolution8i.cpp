/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdUpdate.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        SynetMergedConvolution8i::SynetMergedConvolution8i(const MergConvParam8i& p)
           :  _param(p)
#if defined(SIMD_PERFORMANCE_STATISTIC)
           , _perf(NULL)
#endif        
        {
            const SimdConvolutionParameters& beg = p.conv[0];
            const SimdConvolutionParameters& end = p.conv[p.count - 1];
            _sizeS = beg.srcH * beg.srcW * beg.srcC;
            _sizeD = end.dstH * end.dstW * end.dstC;
            _sizeB[0] = p.conv[1].srcH * p.conv[1].srcW * p.conv[1].srcC;
            _sizeB[1] = p.count == 3 ? p.conv[1].dstH * p.conv[1].dstW * p.conv[1].dstC : 0;
            _src8u = beg.srcT == SimdTensorData8u;
            _dst8u = end.dstT == SimdTensorData8u;
            _dw0 = beg.group != 1;
            switch (p.conv[_dw0 ? 0 : 1].activation)
            {
            case SimdConvolutionActivationIdentity: _depthwise = DepthwiseConvolution<SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: _depthwise = DepthwiseConvolution<SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: _depthwise = DepthwiseConvolution<SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: _depthwise = DepthwiseConvolution<SimdConvolutionActivationHswish>; break;
            default: assert(0);
            }
        }

        size_t SynetMergedConvolution8i::ExternalBufferSize() const
        {
            return 0;
        }

        size_t SynetMergedConvolution8i::InternalBufferSize() const
        {
            size_t size = _buffer.RawSize() + _weight32f.RawSize();
            for (size_t i = 0; i < 3; ++i)
            {
                if (i < 2) 
                    size += _norm[i].RawSize() + _weight8i[i].RawSize();
                size += _bias[i].RawSize() + _params[i].RawSize() + _cvt[i].Size();
            }
            return size;
        }

        void SynetMergedConvolution8i::SetParams(const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params, const float* const* stats)
        {
            const MergConvParam8i& p = _param;
            for (size_t i = 0, ci = 0; i < p.count; ++i)
            {
                const SimdConvolutionParameters& c = p.conv[i];
                if (p.conv[i].group == 1)
                {
                    _weight8i[ci].Resize(c.dstC * c.kernelY * c.kernelX * c.srcC);
                    _norm[ci].Resize(c.dstC);
                    _bias[i].Resize(c.dstC);
                    ci++;
                }
                else
                {
                    _weight32f.Assign(weight[i], c.dstC * c.kernelY * c.kernelX);
                    _bias[i].Assign(bias[i], c.dstC);
                }
                _params[i].Assign(params[i], c.dstC);
                if (internal)
                    internal[i] = SimdTrue;
            }
        }

        void SynetMergedConvolution8i::Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst)
        {
            const MergConvParam8i& p = _param;
        }

#if defined(SIMD_PERFORMANCE_STATISTIC)
        Base::PerformanceMeasurer* SynetMergedConvolution8i::Perf(const String& func)
        {
            if (_perf == NULL)
                _perf = Simd::Base::PerformanceMeasurerStorage::s_storage.Get(func, Param().Info() + " " + Desc(), Param().Flop());
            return _perf;
        }
#endif

        uint8_t* SynetMergedConvolution8i::GetBuffer(uint8_t* buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

        //---------------------------------------------------------------------

        void * SynetMergedConvolution8iInit(size_t batch, const SimdConvolutionParameters * convs, size_t count, SimdSynetCompatibilityType compatibility)
        {
            MergConvParam8i param(batch, convs, count, compatibility);
            if (!param.Valid())
                return NULL;
            return new Base::SynetMergedConvolution8i(param);
        }
    }
}

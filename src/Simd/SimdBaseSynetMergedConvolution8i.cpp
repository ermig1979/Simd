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
        }

        size_t SynetMergedConvolution8i::ExternalBufferSize() const
        {
            return 0;
        }

        size_t SynetMergedConvolution8i::InternalBufferSize() const
        {
            size_t size = _buffer.size;
            return size;
        }

        void SynetMergedConvolution8i::SetParams(const float* const* weight, SimdBool* internal, const float* const* bias, const float* const* params, const float* const* stats)
        {
            const MergConvParam8i& p = _param;
            for (size_t i = 0; i < p.count; ++i)
            {

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

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        typedef Simd::DynamicQuantizedInnerProductParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProductGemmV0::AlgParam AlgParam;
        //typedef Base::SynetQuantizedInnerProductGemmV1::PrepPtr PrepPtr;

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProductGemmV0::SynetDynamicQuantizedInnerProductGemmV0(const DynamicQuantizedInnerProductParam& p)
            : SynetDynamicQuantizedInnerProduct(p)
        {
        }

        String SynetDynamicQuantizedInnerProductGemmV0::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::GemmV0";
            return desc.str();
        }

        size_t SynetDynamicQuantizedInnerProductGemmV0::ExternalBufferSize() const
        {
            size_t size = SynetDynamicQuantizedInnerProduct::ExternalBufferSize();
            size += _sizeA;
            size += _aN * sizeof(int32_t);
            size += _aN * sizeof(float);
            return size;
        }

        void SynetDynamicQuantizedInnerProductGemmV0::Forward(const float* A, uint8_t* buf, float* C)
        {
            const DynamicQuantizedInnerProductParam& p = _param;
            buf = Buffer(buf);
            uint8_t* bufA = Allocate<uint8_t>(buf, _sizeA);
            int32_t* bias = Allocate<int32_t>(buf, _aN);
            float* norm = Allocate<float>(buf, _aN);
            SetInputScaleZero(A, _sizeA);
            float aNorm = 1.0f / _aScale;
            _synetQuantizeLinear(A, _sizeA, &aNorm, _aZero, bufA);
            for (size_t j = 0; j < p.N; ++j)
            {
                bias[j] = _sums[j] * _aZero;
                norm[j] = _scale[j] * _aScale;
            }
#if defined(__MINGW32__) || defined(__MINGW64__)
            bool overflow = true;
#else
            bool overflow = SimdCpuInfo(SimdCpuInfoAvx512vnni) == 0;
#endif
        }

        void SynetDynamicQuantizedInnerProductGemmV0::SetWeight(const int8_t* weight)
        {
            const DynamicQuantizedInnerProductParam& p = _param;
            _weight.Resize(p.N * p.K);
            _weight.Assign(weight, _weight.size);
        }

    }
#endif
}

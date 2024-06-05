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
#include "Simd/SimdSynetInnerProduct16b.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetInnerProduct16bRef::SynetInnerProduct16bRef(const InnerProductParam16b& p)
            :SynetInnerProduct16b(p)
        {
            _sizeA = p.typeA == SimdTensorData32f ? p.M * p.K : 0;
            _sizeB = (p.typeB == SimdTensorData32f && !p.constB) ? p.K * p.N : 0;
            _sizeC = p.typeB == SimdTensorData16b ? p.M * p.N : 0;
        }

        String SynetInnerProduct16bRef::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::Ref";
            return desc.str();
        }

        size_t SynetInnerProduct16bRef::ExternalBufferSize() const
        {
            return _sizeA * 2 + _sizeB * 2 + _sizeC * 4;
        }
        
        void SynetInnerProduct16bRef::SetParams(const float* weight, const float* bias)
        {
            const InnerProductParam16b& p = _param;
            if (p.constB)
            {
                assert(weight);
                _weight.Resize(p.K * p.N);
                Float32ToBFloat16(weight, p.K * p.N, _weight.data);
            }
            _bias.Assign(p.bias ? bias : NULL, p.N);
        }

        void SynetInnerProduct16bRef::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {
            const InnerProductParam16b& p = _param;
            buf = Buffer(buf);
            uint16_t* bufA = (uint16_t*)A;
            if (_sizeA)
            {
                bufA = Allocate<uint16_t>(buf, _sizeA);
                Float32ToBFloat16((float*)A, _sizeA, bufA);
            }
            uint16_t* bufB = (uint16_t*)B;
            if (_sizeB)
            {
                bufB = Allocate<uint16_t>(buf, _sizeB);
                Float32ToBFloat16((float*)B, _sizeB, bufB);
            }
            else if (p.constB)
                bufB = _weight.data;
            float* bufC = (float*)C;
            if (_sizeC)
                bufC = Allocate<float>(buf, _sizeC);
            GemmAndBias(bufA, bufB, bufC);
            if (_sizeC)
                Float32ToBFloat16(bufC, _sizeC, (uint16_t*)C);
        }

        void SynetInnerProduct16bRef::GemmAndBias(const uint16_t* A, const uint16_t* B, float* C)
        {
            const InnerProductParam16b& p = _param;
            for (size_t i = 0; i < p.M; ++i)
            {            
                float* pC = C + i * p.N;
                if (p.transB)
                {
                    for (size_t j = 0; j < p.N; ++j)
                    {
                        const uint16_t* pA = A + i * p.K;
                        const uint16_t* pB = B + j * p.K;
                          for (size_t k = 0; k < p.K; ++k)
                            pC[j] += BFloat16ToFloat32(pA[k]) * BFloat16ToFloat32(pB[k]);
                    }
                }
                else
                {
                    for (size_t j = 0; j < p.N; ++j)
                        pC[j] = 0.0;
                    for (size_t k = 0; k < p.K; ++k)
                    {
                        const uint16_t* pB = B + k * p.N;
                        float a = BFloat16ToFloat32(A[i * p.K + k]);
                        for (size_t j = 0; j < p.N; ++j)
                            pC[j] += a * BFloat16ToFloat32(pB[j]);
                    }
                }
                for (size_t j = 0; j < p.N; ++j)
                    pC[j] += _bias[j];
            }        
        }

        //-------------------------------------------------------------------------------------------------

        void* SynetInnerProduct16bInit(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC, SimdBool transB, SimdBool constB, SimdBool bias)
        {
            InnerProductParam16b param(M, N, K, typeA, typeB, typeC, transB, constB, bias);
            if (!param.Valid())
                return NULL;
            return new SynetInnerProduct16bRef(param);
        }
    }
#endif
}

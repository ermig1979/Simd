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
#include "Simd/SimdSynetQuantizedInnerProduct.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        SynetQuantizedInnerProductGemmNN::SynetQuantizedInnerProductGemmNN(const QuantizedInnerProductParam& p)
            : SynetQuantizedInnerProduct(p)
            , _prepA(0)
            , _prepB(0)
            , _gemm(0)
        {
        }

        String SynetQuantizedInnerProductGemmNN::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::GemmNN";
            return desc.str();
        }

        bool SynetQuantizedInnerProductGemmNN::Preferable(const QuantizedInnerProductParam& p)
        {
            return true;
        }

        void SynetQuantizedInnerProductGemmNN::SetAlgParam(size_t F, size_t microM, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3)
        {
            const QuantizedInnerProductParam& p = _param;
            AlgParam& a = _alg;

            a.F = F;
            a.microM = microM;
            a.microN = microN;
            a.microK = microK;
            a.aK = AlignHi(p.K, a.microK);
            a.aN = AlignHi(p.N, a.F);
            a.aM = AlignHi(p.M, a.microM);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microN, a.microK), a.microK, a.aK);
            a.macroN = Simd::RestrictRange(AlignLo(L3 / a.macroK, a.microN), a.microN, a.aN);
            a.macroM = Simd::RestrictRange(AlignLo(L2 / a.macroK, a.microM), a.microM, a.aM);
            a.eA = p.typeA == SimdTensorData32f ? 4 : 1;
            a.eB = p.typeB == SimdTensorData32f ? 4 : 1;
            a.eC = p.typeC == SimdTensorData32f ? 4 : 1;

            _sizeA = (p.typeA == SimdTensorData32f || p.K != a.aK) ? a.aM * a.aK : 0;
            _sizeB = p.constB ? 0 : a.macroK * a.macroN;
            _sizeC = (p.typeC == SimdTensorData16b || a.aM != p.M || a.aN != p.N) ? a.macroN * a.aM : 0;

            a.bK = p.constB ? a.aK : a.macroK;
            a.cN = _sizeC ? a.macroN : p.N;

            _norm.Resize(a.aN, true);
            _bias.Resize(a.aN, true);
        }

        //size_t SynetQuantizedInnerProductRef::ExternalBufferSize() const
        //{
        //    size_t size = SynetQuantizedInnerProduct::ExternalBufferSize();
        //    if (!_a8u)
        //        size += _sizeA;
        //    size += _sizeC * sizeof(int32_t);
        //    return size;
        //}

        void SynetQuantizedInnerProductGemmNN::SetB(const int8_t* b)
        {
            const QuantizedInnerProductParam& p = _param;
            const AlgParam& a = _alg;
            if (p.constB)
            {
                assert(b);
                _b.Resize(a.aK * a.aN, true);
                _prepB((uint8_t*)b, _bScale[0], 0, p, a, _b.data);
            }
        }

        void SynetQuantizedInnerProductGemmNN::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {
            const QuantizedInnerProductParam& p = _param;
            buf = Buffer(buf);
            //uint8_t* bufA = (uint8_t*)A;
            //if (!_a8u)
            //{
            //    bufA = Allocate<uint8_t>(buf, _sizeA);
            //    SynetQuantizeLinear((float*)A, _sizeA, &_aScale, _aZero[0], bufA);
            //}
            //const int8_t* bufB = (int8_t*)B;
            //if (p.constB)
            //    bufB = _b.data;
            //int32_t* bufC = Allocate<int32_t>(buf, _sizeC);
            //Gemm(bufA, bufB, bufC);
            //if (_c8u)
            //    QuantizeSumLinear(bufC, 1, p.N, 1, p.M, SimdTensorFormatNhwc, _bias.data, _norm.data, _cZero.data, C);
            //else
            //    assert(0);
        }
    }
#endif
}

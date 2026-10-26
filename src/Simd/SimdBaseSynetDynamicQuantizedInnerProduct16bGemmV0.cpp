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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct16b.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        typedef Simd::DynamicQuantizedInnerProduct16bParam DqipParam;
        typedef Base::SynetDynamicQuantizedInnerProduct16bGemmV0::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProduct16bGemmV0::SynetDynamicQuantizedInnerProduct16bGemmV0(const DynamicQuantizedInnerProduct16bParam& p)
            : SynetDynamicQuantizedInnerProduct16b(p)
            , _convertA(NULL)
            , _convertB(NULL)
            , _gemmBody(NULL)
            , _gemmLast(NULL)
        {
        }

        String SynetDynamicQuantizedInnerProduct16bGemmV0::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::GemmV0";
            return desc.str();
        }

        size_t SynetDynamicQuantizedInnerProduct16bGemmV0::ExternalBufferSize() const
        {
            const AlgParam& a = _alg;
            size_t size = SynetDynamicQuantizedInnerProduct16b::ExternalBufferSize();
            size += AlignHi(_sizeA * sizeof(uint8_t), SIMD_ALIGN);
            size += AlignHi(a.aM * sizeof(int32_t), SIMD_ALIGN);
            size += AlignHi(a.aM * sizeof(float), SIMD_ALIGN);
            if (_sizeB)
            {
                size += AlignHi(_sizeB * sizeof(int8_t), SIMD_ALIGN);
                size += AlignHi(a.macroN * sizeof(int32_t), SIMD_ALIGN);
                size += AlignHi(a.macroN * sizeof(float), SIMD_ALIGN);
            }
            size += AlignHi(_sizeC * sizeof(int32_t), SIMD_ALIGN);
            size += AlignHi(_sizeT * sizeof(int32_t), SIMD_ALIGN);
            return size;
        }

        void SynetDynamicQuantizedInnerProduct16bGemmV0::SetWeight(const int8_t* weight)
        {
            const DqipParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.aN * a.aK);
            int8_t* dw = _weight.data;
            size_t N = DivHi(p.N, a.F);
            for (size_t n = 0; n < N; n++)
            {
                for (size_t k = 0; k < a.aK; k += 4)
                {
                    const int8_t* sw = weight + k * p.N + n * a.F;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 4; ++i)
                        {
                            if (n * a.F + f < p.N && k + i < p.K)
                                *(dw++) = sw[i * p.N];
                            else
                                *(dw++) = 0;
                        }
                        sw++;
                    }
                }
            }
        }

        void SynetDynamicQuantizedInnerProduct16bGemmV0::QuantizeB(const float* weight)
        {
            const DqipParam& p = _param;
            const AlgParam& a = _alg;
            _weight.Resize(a.aN * a.aK);
            _convertB((uint8_t*)weight, p, a, p.N, _weight.data, _scale.data, _sums.data);
        }

        void SynetDynamicQuantizedInnerProduct16bGemmV0::SetAlgParam(size_t F, size_t microM, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3)
        {
            const DqipParam& p = _param;
            AlgParam& a = _alg;

            a.F = F;
            a.microM = microM;
            a.microN = microN;
            a.microK = microK;
            a.aK = AlignHiAny(p.K, a.microK);
            a.aN = AlignHiAny(p.N, a.F);
            a.aM = a.microK == 64 ? AlignHiAny(p.M, a.microM) : p.M;
            a.macroK = Simd::RestrictRange(AlignLoAny(L1 / a.microN, a.microK), a.microK, a.aK);
            a.macroN = Simd::RestrictRange(AlignLoAny(L3 / a.macroK, a.microN), a.microN, a.aN);
            a.macroM = Simd::RestrictRange(AlignLoAny(L2 / a.macroK, a.microM), a.microM, a.aM);
            a.eA = p.typeA == SimdTensorData32f ? 4 : 2;
            a.eB = p.typeB == SimdTensorData32f ? 4 : 2;
            a.eC = p.typeC == SimdTensorData32f ? 4 : 2;

            _sizeA = a.aM * a.aK;
            _sizeB = p.constB ? 0 : a.macroN * a.aK;
            _sizeC = a.aK > a.macroK && (p.N != a.aN || p.M != a.aM || a.eC != 4) ? a.macroN * a.aM : 0;
            _sizeT = a.microK == 64 ? 2048 : 0;
            _aN = a.aN;

            a.cN = _sizeC ? a.macroN : p.N;
            a.reorder = 0;
        }

        void SynetDynamicQuantizedInnerProduct16bGemmV0::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {
            const DqipParam& p = _param;
            const AlgParam& a = _alg;
            buf = Buffer(buf);
            uint8_t* bufA = Allocate<uint8_t>(buf, _sizeA);
            float* aScale = Allocate<float>(buf, a.aM);
            int32_t* aZero = Allocate<int32_t>(buf, a.aM);
            int8_t* bufB = _sizeB ? Allocate<int8_t>(buf, _sizeB) : _weight.data;
            float* bScale = _sizeB ? Allocate<float>(buf, a.macroN) : (float*)_scale.data;
            int32_t* bSums = _sizeB ? Allocate<int32_t>(buf, a.macroN) : (int32_t*)_sums.data;
            int32_t* bufC = _sizeC ? Allocate<int32_t>(buf, _sizeC) : (int32_t*)C;
            int32_t* bufT = _sizeT ? Allocate<int32_t>(buf, _sizeT) : (int32_t*)C;
            const float* bias = _bias.data, * params = _params.data;
            for (size_t j = 0; j < p.N; j += a.macroN)
            {
                size_t macroN = Simd::Min(p.N, j + a.macroN) - j;
                if (_sizeB)
                    _convertB(B + j * a.eB, p, a, macroN, bufB, bScale, bSums);
                for (size_t k = 0; k < p.K; k += a.macroK)
                {
                    size_t macroK = Simd::Min(p.K, k + a.macroK) - k;
                    for (size_t i = 0; i < p.M; i += a.macroM)
                    {
                        size_t macroM = Simd::Min(p.M, i + a.macroM) - i;
                        size_t offsA = i * a.aK + (a.reorder ? k * a.F : k);
                        size_t offsC = _sizeC ? i * a.cN : i * a.cN + j;
                        if (j == 0 && k == 0)
                            _convertA(A + i * p.K * a.eA, p, a, macroM, bufA + offsA, aScale + i, aZero + i);
                        if (k + macroK == p.K)
                            _gemmLast(bufA + offsA, p, a, macroM, macroN, macroK, k ? 1 : 0, bufB + k * a.F, bufC + offsC, bufT,
                                aZero + i, bSums, aScale + i, bScale, bias, params, C + (i * p.N + j) * a.eC);
                        else
                            _gemmBody(bufA + offsA, p, a, macroM, macroN, macroK, k ? 1 : 0, bufB + k * a.F, bufC + offsC);
                    }
                }
                bias += macroN;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroN;
                if (p.constB)
                    bScale += macroN, bSums += macroN, bufB += macroN * a.aK;
            }
        }

        bool SynetDynamicQuantizedInnerProduct16bGemmV0::Preferable(const DynamicQuantizedInnerProduct16bParam& p)
        {
            return true;
        }
    }
#endif
}

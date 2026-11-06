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
#include "Simd/SimdSynetDynamicQuantizedInnerProduct32f.h"
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
        typedef Base::SynetDynamicQuantizedInnerProduct32fGemmV0::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        void SynetDynamicQuantizedInnerProduct32fGemmV0_Convert(const float* src, float norm, uint8_t zero, const DqipParam& p, const AlgParam& a, size_t N, uint8_t* dst)
        {
            int min = 0, max = 255;
            size_t K = p.K, KA = a.aK;
            for (size_t j = 0; j < N; ++j)
            {
                size_t k = 0;
                for (; k < K; ++k)
                    dst[k] = QuantizeLinear(src[k], norm, zero, min, max);
                for (; k < KA; ++k)
                    dst[k] = 0;
                src += K;
                dst += KA;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetDynamicQuantizedInnerProduct32fGemmV0::SynetDynamicQuantizedInnerProduct32fGemmV0(const DynamicQuantizedInnerProductParam& p)
            : SynetDynamicQuantizedInnerProduct32f(p)
            , _convert(SynetDynamicQuantizedInnerProduct32fGemmV0_Convert)
            , _gemmBody(NULL)
            , _gemmLast(NULL)
        {
        }

        String SynetDynamicQuantizedInnerProduct32fGemmV0::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::GemmV0";
            return desc.str();
        }

        size_t SynetDynamicQuantizedInnerProduct32fGemmV0::ExternalBufferSize() const
        {
            size_t size = SynetDynamicQuantizedInnerProduct32f::ExternalBufferSize();
            size += _sizeA;
            size += _aN * sizeof(int32_t);
            size += _aN * sizeof(float);
            size += _sizeS * sizeof(int32_t);
            size += _sizeB * sizeof(int32_t);
            return size;
        }

        bool SynetDynamicQuantizedInnerProduct32fGemmV0::Preferable(const DynamicQuantizedInnerProductParam& p)
        {
            return true;
        }

        void SynetDynamicQuantizedInnerProduct32fGemmV0::SetWeight(const int8_t* weight)
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

        void SynetDynamicQuantizedInnerProduct32fGemmV0::SetAlgParam(size_t F, size_t microM, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3)
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

            _sizeA = a.aM * a.aK;
            _sizeS = a.aK > a.macroK && (p.N != a.aN || p.M != a.aM) ? a.macroN * a.aM : 0;
            _sizeB = a.microK == 64 ? 2048 : 0;
            _aN = a.aN;

            a.sN = _sizeS ? a.macroN : p.N;
            a.reorder = 0;
        }

        void SynetDynamicQuantizedInnerProduct32fGemmV0::Forward(const float* A, uint8_t* buf, float* C)
        {
            const DqipParam& p = _param;
            const AlgParam& a = _alg;
            buf = Buffer(buf);
            uint8_t* bufA = Allocate<uint8_t>(buf, _sizeA);
            int32_t* corr = Allocate<int32_t>(buf, _aN);
            float* norm = Allocate<float>(buf, _aN);
            int32_t* bufS = _sizeS ? Allocate<int32_t>(buf, _sizeS) : (int32_t*)C;
            int32_t* bufB = Allocate<int32_t>(buf, _sizeB);
            const float* bias = _bias.data, * params = _params.data;
            AnalyseInput(A, p.M * p.K, corr, norm);
            float aNorm = 1.0f / _aScale;
            for (size_t j = 0; j < p.N; j += a.macroN)
            {
                size_t macroN = Simd::Min(p.N, j + a.macroN) - j;
                for (size_t k = 0; k < p.K; k += a.macroK)
                {
                    size_t macroK = Simd::Min(p.K, k + a.macroK) - k;
                    for (size_t i = 0; i < p.M; i += a.macroM)
                    {
                        size_t macroM = Simd::Min(p.M, i + a.macroM) - i;
                        size_t offsA = i * a.aK + (a.reorder ? k * a.F : k);
                        const int8_t * weight = _weight.data + j * a.aK + k * a.F;
                        size_t offsS = _sizeS ? i * a.sN : i * a.sN + j;
                        if (j == 0 && k == 0)
                            _convert(A + i * p.K, aNorm, _aZero, p, a, macroM, bufA + offsA);
                        if (k + macroK == p.K)
                            _gemmLast(bufA + offsA, p, a, macroM, macroN, macroK, k ? 1 : 0, weight, bufS + offsS, bufB,
                                corr + j, norm + j, bias, params, C + i * p.N + j);
                        else
                            _gemmBody(bufA + offsA, p, a, macroM, macroN, macroK, k ? 1 : 0, weight, bufS + offsS);
                    }
                }
                bias += macroN;
                if (p.activation == ::SimdConvolutionActivationPrelu)
                    params += macroN;
            }
        }
    }
#endif
}

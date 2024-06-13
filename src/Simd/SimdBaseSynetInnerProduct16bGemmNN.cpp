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
        typedef Base::SynetInnerProduct16bGemmNN::AlgParam AlgParam;

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ConvertBn(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t Kh = AlignHi(K, a.microK), gap = (a.bK - Kh) * a.F;
            for (size_t j = 0; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const float* ps = src + k * p.N + j;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < p.N && k + i < p.K)
                                *(dst++) = Float32ToBFloat16(ps[i * p.N + f]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
        }

        //-----------------------------------------------------------------------------------------

        static void InnerProduct16bGemmNN_ConvertBt(const uint8_t* src8, const InnerProductParam16b& p, const AlgParam& a, size_t N, size_t K, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t Kh = AlignHi(K, a.microK), gap = (a.bK - Kh) * a.F;
            for (size_t j = 0; j < N; j += a.F)
            {
                for (size_t k = 0; k < Kh; k += 2)
                {
                    const float* ps = src + j * p.K + k;
                    for (size_t f = 0; f < a.F; ++f)
                    {
                        for (size_t i = 0; i < 2; ++i)
                        {
                            if (j + f < p.N && k + i < p.K)
                                *(dst++) = Float32ToBFloat16(ps[f * p.K + i]);
                            else
                                *(dst++) = 0;
                        }
                    }
                }
                dst += gap;
            }
        }

        //-----------------------------------------------------------------------------------------

        bool SynetInnerProduct16bGemmNN::Preferable(const InnerProductParam16b& p)
        {
            return true;// p.constB == SimdTrue || p.typeB == SimdTensorData32f;
        }

        SynetInnerProduct16bGemmNN::SynetInnerProduct16bGemmNN(const InnerProductParam16b& p)
            : SynetInnerProduct16b(p)
            , _alg({ 0 })
            , _prepA(0)
            , _prepB(0)
            , _gemm(0)
        {
            if (p.typeB == SimdTensorData32f || p.constB)
            {
                if (p.transB)
                    _prepB = InnerProduct16bGemmNN_ConvertBt;
                else
                    _prepB = InnerProduct16bGemmNN_ConvertBn;
            }
        }

        String SynetInnerProduct16bGemmNN::Desc() const
        {
            std::stringstream desc;
            desc << Ext() << "::GemmNN";
            return desc.str();
        }

        void SynetInnerProduct16bGemmNN::SetAlgParam(size_t F, size_t microM, size_t microN, size_t microK, size_t L1, size_t L2, size_t L3)
        {
            const InnerProductParam16b& p = _param;
            AlgParam& a = _alg;

            a.F = F;
            a.microM = microM;
            a.microN = microN;
            a.microK = microK;
            a.aK = AlignHi(p.K, a.microK);
            a.aN = AlignHi(p.N, a.F);
            a.aM = AlignHi(p.M, a.microM);
            a.macroK = Simd::RestrictRange(AlignLo(L1 / a.microN / 2, a.microK), a.microK, a.aK);
            a.macroN = Simd::RestrictRange(AlignLo(L3 / a.macroK / 2, a.microN), a.microN, a.aN);
            a.macroM = Simd::RestrictRange(L2 / a.macroK / 2, a.microM, a.aM);
            a.eA = p.typeA == SimdTensorData32f ? 4 : 2;
            a.eB = p.typeB == SimdTensorData32f ? 4 : 2;
            a.eC = p.typeC == SimdTensorData32f ? 4 : 2;
            a.bK = p.constB ? a.aK : a.macroK;
            a.cN = p.typeC == SimdTensorData32f || a.macroK < a.aK ? p.N : a.macroN;

            _sizeA = (p.typeA == SimdTensorData32f || p.K != a.aK) ? (a.macroN == a.aN ? a.macroM : a.aM) * a.aK : 0;
            _sizeB = p.constB ? 0 : a.macroK * a.macroN;
            _sizeC = p.typeC == SimdTensorData16b ? a.macroM * a.cN : 0;
        }

        void SynetInnerProduct16bGemmNN::SetParams(const float* weight, const float* bias)
        {
            const InnerProductParam16b& p = _param;
            const AlgParam& a = _alg;
            if (p.constB)
            {
                assert(weight);
                _weight.Resize(a.aK * a.aN, true);
                _prepB((uint8_t*)weight, p, a, p.N, p.K, _weight.data);
            }
            _bias.Resize(a.aN, true);
            if (p.bias && bias)
                memcpy(_bias.data, bias, p.N * 4);
        }

        void SynetInnerProduct16bGemmNN::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {
            const InnerProductParam16b& p = _param;
            const AlgParam& a = _alg;
            buf = Buffer(buf);
            uint16_t* bufA = _prepA ? Allocate<uint16_t>(buf, _sizeA) : (uint16_t*)A;
            uint16_t* bufB = p.constB ? _weight.data : Allocate<uint16_t>(buf, _sizeB);
            float* bufC = _sizeC ? Allocate<float>(buf, _sizeC) : (float*)C;
            for (size_t j = 0; j < p.N; j += a.macroN)
            {
                size_t macroN = Simd::Min(p.N, j + a.macroN) - j;
                for (size_t k = 0; k < p.K; k += a.macroK)
                {
                    size_t macroK = Simd::Min(p.K, k + a.macroK) - k;
                    for (size_t i = 0; i < p.M; i += a.macroM)
                    {
                        size_t macroM = Simd::Min(p.M, i + a.macroM) - i;
                        size_t offsA = (a.macroN == a.aN && _prepA) ? 0 : i * a.aK;
                        size_t offsB = p.constB ? j * a.bK + k * a.F : 0;
                        size_t offsC = _sizeC ? 0 : i * a.cN + j;
                        if (j == 0 && k == 0 && _prepA)
                            _prepA(A + i * p.K * a.eA, p, a, macroM, p.K, bufA + offsA);
                        if (i == 0 && _prepB && !p.constB)
                            _prepB(B + (p.transB ? j * p.K + k : k * p.N + j) * a.eB, p, a, macroN, macroK, bufB + offsB);
                        _gemm(bufA + offsA + k, p, a, macroM, macroN, macroK, (int)k, bufB + offsB, bufC + offsC, 
                            k + macroK == p.K && (_sizeC || p.bias), _bias.data + j, C + (i * p.N + j) * a.eC);
                    }
                }
            }
        }
    }
#endif
}

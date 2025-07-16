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
        typedef Simd::QuantizedInnerProductParam QipParam;
        typedef Base::SynetQuantizedInnerProductGemmNN::AlgParam AlgParam;
        typedef Base::SynetQuantizedInnerProductGemmNN::PrepPtr PrepPtr;
        typedef Base::SynetQuantizedInnerProductGemmNN::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        static void SynetQuantizedInnerProductGemmNN_PrepB_8i(const uint8_t* src, float norm, uint8_t zero, const QipParam& p, const AlgParam& a, size_t, size_t, uint8_t* dst)
        {
            size_t N = DivHi(p.N, a.F);
            for (size_t n = 0; n < N; n++)
            {
                for (size_t k = 0; k < a.aK; k += 4)
                {
                    if (p.transB)
                    {
                        const uint8_t* ps = src + n * a.F * p.K + k;
                        for (size_t f = 0; f < a.F; ++f)
                        {
                            for (size_t i = 0; i < 4; ++i)
                            {
                                if (n * a.F + f < p.N && k + i < p.K)
                                    *(dst++) = ps[i];
                                else
                                    *(dst++) = 0;
                            }
                            ps += p.K;
                        }
                    }
                    else
                    {
                        const uint8_t* ps = src + k * p.N + n * a.F;
                        for (size_t f = 0; f < a.F; ++f)
                        {
                            for (size_t i = 0; i < 4; ++i)
                            {
                                if (n * a.F + f < p.N && k + i < p.K)
                                    *(dst++) = ps[i * p.N];
                                else
                                    *(dst++) = 0;
                            }
                            ps++;
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------


        SynetQuantizedInnerProductGemmNN::SynetQuantizedInnerProductGemmNN(const QuantizedInnerProductParam& p)
            : SynetQuantizedInnerProduct(p)
            , _prepA(0)
            , _prepB(0)
            , _gemm(0)
        {
            _prepB = SynetQuantizedInnerProductGemmNN_PrepB_8i;
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
            _sizeC = a.macroN * a.aM;
            _aN = a.aN;

            a.bK = p.constB ? a.aK : a.macroK;
            a.cN = a.macroN;

            _norm.Resize(a.aN, true);
            _bias.Resize(a.aN, true);
        }

        size_t SynetQuantizedInnerProductGemmNN::ExternalBufferSize() const
        {
            const QuantizedInnerProductParam& p = _param;
            const AlgParam& a = _alg;
            size_t size = SynetQuantizedInnerProduct::ExternalBufferSize();
            if (!_a8u || a.aK != p.K)
                size += _sizeA;
            size += _sizeC * sizeof(int32_t);
            return size;
        }

        void SynetQuantizedInnerProductGemmNN::SetB(const int8_t* b)
        {
            const QuantizedInnerProductParam& p = _param;
            const AlgParam& a = _alg;
            if (p.constB)
            {
                assert(b);
                _b.Resize(a.aK * a.aN, true);
                _prepB((uint8_t*)b, _bScale[0], 0, p, a, p.N, p.K, (uint8_t*)_b.data);
            }
        }

        void SynetQuantizedInnerProductGemmNN::Forward(const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C)
        {
            const QuantizedInnerProductParam& p = _param;
            const AlgParam& a = _alg;
            buf = Buffer(buf);
            uint8_t* bufA = _prepA ? Allocate<uint8_t>(buf, _sizeA) : (uint8_t*)A;
            int8_t* bufB = p.constB ? _b.data : Allocate<int8_t>(buf, _sizeB);
            int32_t* bufC = Allocate<int32_t>(buf, _sizeC);
            for (size_t j = 0; j < p.N; j += a.macroN)
            {
                size_t macroN = Simd::Min(p.N, j + a.macroN) - j;
                for (size_t k = 0; k < p.K; k += a.macroK)
                {
                    size_t macroK = Simd::Min(p.K, k + a.macroK) - k;
                    for (size_t i = 0; i < p.M; i += a.macroM)
                    {
                        size_t macroM = Simd::Min(p.M, i + a.macroM) - i;
                        size_t offsA = (a.macroN == a.aN && a.macroK == a.aK && _prepA) ? 0 : i * a.aK;
                        size_t offsB = p.constB ? j * a.bK + k * a.F : 0;
                        size_t offsC = _sizeC ? (a.macroK < a.aK ? i * a.cN : 0) : i * a.cN + j;
                        if (j == 0 && k == 0 && _prepA)
                            _prepA(A + i * p.K * a.eA, _aScale, _aZero[0], p, a, macroM, p.K, bufA + offsA);
                        //if (i == 0 && _prepB && !p.constB)
                        //    _prepB(B + (p.transB ? j * p.K + k : k * p.N + j) * a.eB, p, a, macroN, macroK, bufB + offsB);
                        _gemm(bufA + offsA + k, p, a, macroM, macroN, macroK, (int)k, bufB + offsB, bufC + offsC,
                            k + macroK == p.K && (_sizeC || p.bias), _bias.data + j, _norm.data + j, _cZero[0], C + (i * p.N + j) * a.eC);
                    }
                }
            }
        }
    }
#endif
}
